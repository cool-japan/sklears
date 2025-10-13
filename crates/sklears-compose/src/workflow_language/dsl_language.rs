//! Domain-Specific Language (DSL) for Machine Learning Pipelines
//!
//! This module provides a text-based syntax for defining pipelines in a concise, readable format.
//! Includes lexical analysis, parsing, and AST generation for the workflow DSL.
//!
//! Example DSL syntax:
//! ```text
//! pipeline "Customer Churn Prediction" {
//!     version "1.0.0"
//!     author "Data Science Team"
//!
//!     input features: Matrix<f64> [samples, features]
//!     input labels: Array<f64> [samples]
//!
//!     step scaler: StandardScaler {
//!         with_mean: true,
//!         with_std: true
//!     }
//!
//!     step model: RandomForestClassifier {
//!         n_estimators: 100,
//!         max_depth: 10,
//!         random_state: 42
//!     }
//!
//!     flow features -> scaler.X
//!     flow scaler.X_scaled -> model.X
//!     flow labels -> model.y
//!
//!     output predictions: model.predictions
//!
//!     execute {
//!         mode: parallel,
//!         workers: 4
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{BTreeMap, VecDeque};

use super::workflow_definitions::{
    Connection, ConnectionType, DataType, ExecutionConfig, ExecutionMode, InputDefinition,
    OutputDefinition, ParallelConfig, ParameterValue, StepDefinition, StepType, WorkflowDefinition,
};

/// Domain-Specific Language (DSL) for machine learning pipelines
#[derive(Debug)]
pub struct PipelineDSL {
    /// DSL lexer
    lexer: DslLexer,
    /// DSL parser
    parser: DslParser,
}

/// DSL Lexer for tokenizing pipeline definitions
#[derive(Debug)]
pub struct DslLexer {
    /// Input text
    input: String,
    /// Cached input characters for efficient indexing
    input_chars: Vec<char>,
    /// Current position
    position: usize,
    /// Current line number
    line: usize,
    /// Current column number
    column: usize,
}

/// DSL Parser for converting tokens to workflow definitions
#[derive(Debug)]
pub struct DslParser {
    /// Token stream
    tokens: VecDeque<Token>,
    /// Current token index
    current: usize,
}

/// Token types for the DSL
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    /// Pipeline
    Pipeline,
    /// Version
    Version,
    /// Author
    Author,
    /// Description
    Description,
    /// Input
    Input,
    /// Output
    Output,
    /// Step
    Step,
    /// Flow
    Flow,
    /// Execute
    Execute,

    // Data types
    /// Matrix
    Matrix,
    /// Array
    Array,
    /// Float32
    Float32,
    /// Float64
    Float64,
    /// Int32
    Int32,
    /// Int64
    Int64,
    /// Bool
    Bool,
    /// String
    String,

    // Literals
    /// Identifier
    Identifier(String),
    /// StringLiteral
    StringLiteral(String),
    /// NumberLiteral
    NumberLiteral(f64),
    /// BooleanLiteral
    BooleanLiteral(bool),

    // Operators and punctuation
    /// LeftBrace
    LeftBrace,
    /// RightBrace
    RightBrace,
    /// LeftBracket
    LeftBracket,
    /// RightBracket
    RightBracket,
    /// LeftParen
    LeftParen,
    /// RightParen
    RightParen,
    /// LeftAngle
    LeftAngle,
    /// RightAngle
    RightAngle,
    /// Comma
    Comma,
    /// Colon
    Colon,
    /// Semicolon
    Semicolon,
    /// Dot
    Dot,
    /// Arrow
    Arrow,

    // Special
    /// Newline
    Newline,
    /// Eof
    Eof,
}

/// DSL parsing error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DslError {
    /// Error message
    pub message: String,
    /// Line number where error occurred
    pub line: usize,
    /// Column number where error occurred
    pub column: usize,
}

impl PipelineDSL {
    /// Create a new DSL processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            lexer: DslLexer::new(),
            parser: DslParser::new(),
        }
    }

    /// Parse DSL text into a workflow definition
    pub fn parse(&mut self, input: &str) -> SklResult<WorkflowDefinition> {
        // Tokenize input
        let tokens = self.lexer.tokenize(input)?;

        // Parse tokens into workflow
        self.parser.parse(tokens)
    }

    /// Generate DSL text from workflow definition
    #[must_use]
    pub fn generate(&self, workflow: &WorkflowDefinition) -> String {
        let mut dsl = String::new();

        // Pipeline header
        dsl.push_str(&format!("pipeline \"{}\" {{\n", workflow.metadata.name));
        dsl.push_str(&format!("    version \"{}\"\n", workflow.metadata.version));

        if let Some(author) = &workflow.metadata.author {
            dsl.push_str(&format!("    author \"{author}\"\n"));
        }

        if let Some(description) = &workflow.metadata.description {
            dsl.push_str(&format!("    description \"{description}\"\n"));
        }

        dsl.push('\n');

        // Inputs
        for input in &workflow.inputs {
            dsl.push_str(&format!(
                "    input {}: {}\n",
                input.name,
                self.format_data_type(&input.data_type)
            ));
        }

        if !workflow.inputs.is_empty() {
            dsl.push('\n');
        }

        // Steps
        for step in &workflow.steps {
            dsl.push_str(&format!("    step {}: {} {{\n", step.id, step.algorithm));

            for (param_name, param_value) in &step.parameters {
                dsl.push_str(&format!(
                    "        {}: {},\n",
                    param_name,
                    self.format_parameter_value(param_value)
                ));
            }

            dsl.push_str("    }\n\n");
        }

        // Connections
        for connection in &workflow.connections {
            dsl.push_str(&format!(
                "    flow {} -> {}\n",
                format!("{}.{}", connection.from_step, connection.from_output),
                format!("{}.{}", connection.to_step, connection.to_input)
            ));
        }

        if !workflow.connections.is_empty() {
            dsl.push('\n');
        }

        // Outputs
        for output in &workflow.outputs {
            dsl.push_str(&format!("    output {}\n", output.name));
        }

        if !workflow.outputs.is_empty() {
            dsl.push('\n');
        }

        // Execution configuration
        if workflow.execution.mode != ExecutionMode::Sequential {
            dsl.push_str("    execute {\n");
            dsl.push_str(&format!("        mode: {:?},\n", workflow.execution.mode));

            if let Some(parallel_config) = &workflow.execution.parallel {
                dsl.push_str(&format!(
                    "        workers: {}\n",
                    parallel_config.num_workers
                ));
            }

            dsl.push_str("    }\n");
        }

        dsl.push_str("}\n");
        dsl
    }

    /// Format data type for DSL output
    fn format_data_type(&self, data_type: &DataType) -> String {
        match data_type {
            DataType::Float32 => "f32".to_string(),
            DataType::Float64 => "f64".to_string(),
            DataType::Int32 => "i32".to_string(),
            DataType::Int64 => "i64".to_string(),
            DataType::Boolean => "bool".to_string(),
            DataType::String => "String".to_string(),
            DataType::Array(inner) => format!("Array<{}>", self.format_data_type(inner)),
            DataType::Matrix(inner) => format!("Matrix<{}>", self.format_data_type(inner)),
            DataType::Custom(name) => name.clone(),
        }
    }

    /// Format parameter value for DSL output
    fn format_parameter_value(&self, value: &ParameterValue) -> String {
        match value {
            ParameterValue::Float(f) => f.to_string(),
            ParameterValue::Int(i) => i.to_string(),
            ParameterValue::Bool(b) => b.to_string(),
            ParameterValue::String(s) => format!("\"{s}\""),
            ParameterValue::Array(arr) => {
                let items: Vec<String> =
                    arr.iter().map(|v| self.format_parameter_value(v)).collect();
                format!("[{}]", items.join(", "))
            }
        }
    }

    /// Validate DSL syntax
    pub fn validate_syntax(&mut self, input: &str) -> SklResult<Vec<DslError>> {
        let mut errors = Vec::new();

        // Attempt to tokenize
        match self.lexer.tokenize(input) {
            Ok(tokens) => {
                // Attempt to parse
                match self.parser.parse(tokens) {
                    Ok(_) => {
                        // Syntax is valid
                    }
                    Err(e) => {
                        errors.push(DslError {
                            message: e.to_string(),
                            line: 1, // Parser should provide line/column info
                            column: 1,
                        });
                    }
                }
            }
            Err(e) => {
                errors.push(DslError {
                    message: e.to_string(),
                    line: self.lexer.line,
                    column: self.lexer.column,
                });
            }
        }

        Ok(errors)
    }

    /// Get syntax highlighting information
    pub fn get_syntax_highlighting(&mut self, input: &str) -> Vec<SyntaxHighlight> {
        let mut highlights = Vec::new();

        if let Ok(tokens) = self.lexer.tokenize(input) {
            let mut position = 0;

            for token in tokens {
                let (token_type, length) = match &token {
                    Token::Pipeline
                    | Token::Version
                    | Token::Author
                    | Token::Description
                    | Token::Input
                    | Token::Output
                    | Token::Step
                    | Token::Flow
                    | Token::Execute => ("keyword", self.estimate_token_length(&token)),
                    Token::Matrix
                    | Token::Array
                    | Token::Float32
                    | Token::Float64
                    | Token::Int32
                    | Token::Int64
                    | Token::Bool
                    | Token::String => ("type", self.estimate_token_length(&token)),
                    Token::StringLiteral(_) => ("string", self.estimate_token_length(&token)),
                    Token::NumberLiteral(_) => ("number", self.estimate_token_length(&token)),
                    Token::BooleanLiteral(_) => ("boolean", self.estimate_token_length(&token)),
                    Token::Identifier(_) => ("identifier", self.estimate_token_length(&token)),
                    _ => ("punctuation", self.estimate_token_length(&token)),
                };

                highlights.push(SyntaxHighlight {
                    start: position,
                    end: position + length,
                    token_type: token_type.to_string(),
                });

                position += length;
            }
        }

        highlights
    }

    /// Estimate token length for highlighting
    fn estimate_token_length(&self, token: &Token) -> usize {
        match token {
            Token::Identifier(s) | Token::StringLiteral(s) => s.len(),
            Token::NumberLiteral(n) => n.to_string().len(),
            Token::BooleanLiteral(b) => b.to_string().len(),
            Token::Pipeline => "pipeline".len(),
            Token::Version => "version".len(),
            Token::Author => "author".len(),
            Token::Description => "description".len(),
            Token::Input => "input".len(),
            Token::Output => "output".len(),
            Token::Step => "step".len(),
            Token::Flow => "flow".len(),
            Token::Execute => "execute".len(),
            Token::Matrix => "Matrix".len(),
            Token::Array => "Array".len(),
            Token::Float32 => "f32".len(),
            Token::Float64 => "f64".len(),
            Token::Int32 => "i32".len(),
            Token::Int64 => "i64".len(),
            Token::Bool => "bool".len(),
            Token::String => "String".len(),
            Token::Arrow => "->".len(),
            _ => 1,
        }
    }
}

/// Syntax highlighting information
#[derive(Debug, Clone)]
pub struct SyntaxHighlight {
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Token type for styling
    pub token_type: String,
}

impl DslLexer {
    /// Create a new lexer
    #[must_use]
    pub fn new() -> Self {
        Self {
            input: String::new(),
            input_chars: Vec::new(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    /// Tokenize input text
    pub fn tokenize(&mut self, input: &str) -> SklResult<VecDeque<Token>> {
        self.input = input.to_string();
        self.input_chars = self.input.chars().collect();
        self.position = 0;
        self.line = 1;
        self.column = 1;

        let mut tokens = VecDeque::new();

        while !self.is_at_end() {
            self.skip_whitespace();

            if self.is_at_end() {
                break;
            }

            let token = self.scan_token()?;
            if token != Token::Newline {
                // Skip newlines for now
                tokens.push_back(token);
            }
        }

        tokens.push_back(Token::Eof);
        Ok(tokens)
    }

    /// Scan next token
    fn scan_token(&mut self) -> SklResult<Token> {
        let c = self.advance();

        match c {
            '{' => Ok(Token::LeftBrace),
            '}' => Ok(Token::RightBrace),
            '[' => Ok(Token::LeftBracket),
            ']' => Ok(Token::RightBracket),
            '(' => Ok(Token::LeftParen),
            ')' => Ok(Token::RightParen),
            '<' => Ok(Token::LeftAngle),
            '>' => Ok(Token::RightAngle),
            ',' => Ok(Token::Comma),
            ':' => Ok(Token::Colon),
            ';' => Ok(Token::Semicolon),
            '.' => Ok(Token::Dot),
            '\n' => {
                self.line += 1;
                self.column = 1;
                Ok(Token::Newline)
            }
            '-' => {
                if self.match_char('>') {
                    Ok(Token::Arrow)
                } else {
                    self.scan_number()
                }
            }
            '"' => self.scan_string(),
            _ if c.is_ascii_digit() => {
                self.position -= 1; // Back up to scan full number
                self.column -= 1;
                self.scan_number()
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                self.position -= 1; // Back up to scan full identifier
                self.column -= 1;
                self.scan_identifier()
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unexpected character '{}' at line {}, column {}",
                c, self.line, self.column
            ))),
        }
    }

    /// Scan string literal
    fn scan_string(&mut self) -> SklResult<Token> {
        let mut value = String::new();

        while !self.is_at_end() && self.peek() != '"' {
            if self.peek() == '\n' {
                self.line += 1;
                self.column = 1;
            }
            value.push(self.advance());
        }

        if self.is_at_end() {
            return Err(SklearsError::InvalidInput(format!(
                "Unterminated string at line {}",
                self.line
            )));
        }

        // Consume closing quote
        self.advance();

        Ok(Token::StringLiteral(value))
    }

    /// Scan number literal
    fn scan_number(&mut self) -> SklResult<Token> {
        let mut value = String::new();

        // Handle negative numbers
        if self.peek() == '-' {
            value.push(self.advance());
        }

        while !self.is_at_end() && (self.peek().is_ascii_digit() || self.peek() == '.') {
            value.push(self.advance());
        }

        match value.parse::<f64>() {
            Ok(number) => Ok(Token::NumberLiteral(number)),
            Err(_) => Err(SklearsError::InvalidInput(format!(
                "Invalid number '{}' at line {}, column {}",
                value, self.line, self.column
            ))),
        }
    }

    /// Scan identifier or keyword
    fn scan_identifier(&mut self) -> SklResult<Token> {
        let mut value = String::new();

        while !self.is_at_end() && (self.peek().is_ascii_alphanumeric() || self.peek() == '_') {
            value.push(self.advance());
        }

        // Check for keywords
        let token = match value.as_str() {
            "pipeline" => Token::Pipeline,
            "version" => Token::Version,
            "author" => Token::Author,
            "description" => Token::Description,
            "input" => Token::Input,
            "output" => Token::Output,
            "step" => Token::Step,
            "flow" => Token::Flow,
            "execute" => Token::Execute,
            "Matrix" => Token::Matrix,
            "Array" => Token::Array,
            "f32" => Token::Float32,
            "f64" => Token::Float64,
            "i32" => Token::Int32,
            "i64" => Token::Int64,
            "bool" => Token::Bool,
            "String" => Token::String,
            "true" => Token::BooleanLiteral(true),
            "false" => Token::BooleanLiteral(false),
            _ => Token::Identifier(value),
        };

        Ok(token)
    }

    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        while !self.is_at_end() {
            match self.peek() {
                ' ' | '\r' | '\t' => {
                    self.advance();
                }
                '/' if self.peek_next() == '/' => {
                    // Line comment
                    while !self.is_at_end() && self.peek() != '\n' {
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    /// Advance to next character
    fn advance(&mut self) -> char {
        if let Some(&c) = self.input_chars.get(self.position) {
            self.position += 1;
            self.column += 1;
            c
        } else {
            '\0'
        }
    }

    /// Peek at current character
    fn peek(&self) -> char {
        self.input_chars.get(self.position).copied().unwrap_or('\0')
    }

    /// Peek at next character
    fn peek_next(&self) -> char {
        self.input_chars
            .get(self.position + 1)
            .copied()
            .unwrap_or('\0')
    }

    /// Check if character matches expected
    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() || self.peek() != expected {
            false
        } else {
            self.advance();
            true
        }
    }

    /// Check if at end of input
    fn is_at_end(&self) -> bool {
        self.position >= self.input_chars.len()
    }
}

impl DslParser {
    /// Create a new parser
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: VecDeque::new(),
            current: 0,
        }
    }

    /// Parse tokens into workflow definition
    pub fn parse(&mut self, tokens: VecDeque<Token>) -> SklResult<WorkflowDefinition> {
        self.tokens = tokens;
        self.current = 0;

        self.parse_pipeline()
    }

    /// Parse pipeline definition
    fn parse_pipeline(&mut self) -> SklResult<WorkflowDefinition> {
        self.consume(Token::Pipeline, "Expected 'pipeline'")?;

        let name = if let Token::StringLiteral(name) = self.advance() {
            name
        } else {
            return Err(SklearsError::InvalidInput(
                "Expected pipeline name".to_string(),
            ));
        };

        self.consume(Token::LeftBrace, "Expected '{' after pipeline name")?;

        let mut workflow = WorkflowDefinition::default();
        workflow.metadata.name = name;

        // Parse pipeline body
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            match &self.peek() {
                Token::Version => {
                    self.advance();
                    if let Token::StringLiteral(version) = self.advance() {
                        workflow.metadata.version = version;
                    }
                }
                Token::Author => {
                    self.advance();
                    if let Token::StringLiteral(author) = self.advance() {
                        workflow.metadata.author = Some(author);
                    }
                }
                Token::Description => {
                    self.advance();
                    if let Token::StringLiteral(description) = self.advance() {
                        workflow.metadata.description = Some(description);
                    }
                }
                Token::Input => {
                    workflow.inputs.push(self.parse_input()?);
                }
                Token::Output => {
                    workflow.outputs.push(self.parse_output()?);
                }
                Token::Step => {
                    workflow.steps.push(self.parse_step()?);
                }
                Token::Flow => {
                    workflow.connections.push(self.parse_flow()?);
                }
                Token::Execute => {
                    workflow.execution = self.parse_execute()?;
                }
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unexpected token: {:?}",
                        self.peek()
                    )));
                }
            }
        }

        self.consume(Token::RightBrace, "Expected '}' after pipeline body")?;

        Ok(workflow)
    }

    /// Parse input definition
    fn parse_input(&mut self) -> SklResult<InputDefinition> {
        self.consume(Token::Input, "Expected 'input'")?;

        let name = if let Token::Identifier(name) = self.advance() {
            name
        } else {
            return Err(SklearsError::InvalidInput(
                "Expected input name".to_string(),
            ));
        };

        self.consume(Token::Colon, "Expected ':' after input name")?;

        let data_type = self.parse_data_type()?;

        Ok(InputDefinition {
            name,
            data_type,
            shape: None,
            constraints: None,
            description: None,
        })
    }

    /// Parse output definition
    fn parse_output(&mut self) -> SklResult<OutputDefinition> {
        self.consume(Token::Output, "Expected 'output'")?;

        let name = if let Token::Identifier(name) = self.advance() {
            name
        } else {
            return Err(SklearsError::InvalidInput(
                "Expected output name".to_string(),
            ));
        };

        Ok(OutputDefinition {
            name,
            data_type: DataType::Float64, // Default type
            shape: None,
            description: None,
        })
    }

    /// Parse step definition
    fn parse_step(&mut self) -> SklResult<StepDefinition> {
        self.consume(Token::Step, "Expected 'step'")?;

        let id = if let Token::Identifier(id) = self.advance() {
            id
        } else {
            return Err(SklearsError::InvalidInput(
                "Expected step identifier".to_string(),
            ));
        };

        self.consume(Token::Colon, "Expected ':' after step identifier")?;

        let algorithm = if let Token::Identifier(algorithm) = self.advance() {
            algorithm
        } else {
            return Err(SklearsError::InvalidInput(
                "Expected algorithm name".to_string(),
            ));
        };

        let mut parameters = BTreeMap::new();

        if self.check(&Token::LeftBrace) {
            self.advance(); // consume '{'

            while !self.check(&Token::RightBrace) && !self.is_at_end() {
                let param_name = if let Token::Identifier(name) = self.advance() {
                    name
                } else {
                    return Err(SklearsError::InvalidInput(
                        "Expected parameter name".to_string(),
                    ));
                };

                self.consume(Token::Colon, "Expected ':' after parameter name")?;

                let param_value = self.parse_parameter_value()?;
                parameters.insert(param_name, param_value);

                if self.check(&Token::Comma) {
                    self.advance();
                }
            }

            self.consume(Token::RightBrace, "Expected '}' after step parameters")?;
        }

        Ok(StepDefinition {
            id,
            step_type: StepType::Custom("DSL".to_string()),
            algorithm,
            parameters,
            inputs: Vec::new(),
            outputs: Vec::new(),
            condition: None,
            description: None,
        })
    }

    /// Parse flow/connection definition
    fn parse_flow(&mut self) -> SklResult<Connection> {
        self.consume(Token::Flow, "Expected 'flow'")?;

        // Parse source (step.output)
        let from_step = if let Token::Identifier(step) = self.advance() {
            step
        } else {
            return Err(SklearsError::InvalidInput(
                "Expected source step".to_string(),
            ));
        };

        self.consume(Token::Dot, "Expected '.' after source step")?;

        let from_output = match self.advance() {
            Token::Identifier(output) => output,
            Token::Output => "output".to_string(),
            Token::Input => "input".to_string(),
            other => {
                return Err(SklearsError::InvalidInput(format!(
                    "Expected source output, got {:?}",
                    other
                )))
            }
        };

        self.consume(Token::Arrow, "Expected '->' in flow")?;

        // Parse target (step.input)
        let to_step = if let Token::Identifier(step) = self.advance() {
            step
        } else {
            return Err(SklearsError::InvalidInput(
                "Expected target step".to_string(),
            ));
        };

        self.consume(Token::Dot, "Expected '.' after target step")?;

        let to_input = match self.advance() {
            Token::Identifier(input) => input,
            Token::Input => "input".to_string(),
            other => {
                return Err(SklearsError::InvalidInput(format!(
                    "Expected target input, got {:?}",
                    other
                )))
            }
        };

        Ok(Connection {
            from_step,
            from_output,
            to_step,
            to_input,
            connection_type: ConnectionType::Direct,
            transform: None,
        })
    }

    /// Parse execution configuration
    fn parse_execute(&mut self) -> SklResult<ExecutionConfig> {
        self.consume(Token::Execute, "Expected 'execute'")?;
        self.consume(Token::LeftBrace, "Expected '{' after 'execute'")?;

        let mut config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            parallel: None,
            resources: None,
            caching: None,
        };

        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            let key = if let Token::Identifier(key) = self.advance() {
                key
            } else {
                return Err(SklearsError::InvalidInput(
                    "Expected configuration key".to_string(),
                ));
            };

            self.consume(Token::Colon, "Expected ':' after configuration key")?;

            match key.as_str() {
                "mode" => {
                    if let Token::Identifier(mode) = self.advance() {
                        config.mode = match mode.as_str() {
                            "parallel" => ExecutionMode::Parallel,
                            "sequential" => ExecutionMode::Sequential,
                            "distributed" => ExecutionMode::Distributed,
                            "gpu" => ExecutionMode::GPU,
                            "adaptive" => ExecutionMode::Adaptive,
                            _ => ExecutionMode::Sequential,
                        };
                    }
                }
                "workers" => {
                    if let Token::NumberLiteral(workers) = self.advance() {
                        config.parallel = Some(ParallelConfig {
                            num_workers: workers as usize,
                            chunk_size: None,
                            load_balancing: "round_robin".to_string(),
                        });
                    }
                }
                _ => {
                    // Skip unknown keys
                    self.advance();
                }
            }

            if self.check(&Token::Comma) {
                self.advance();
            }
        }

        self.consume(
            Token::RightBrace,
            "Expected '}' after execute configuration",
        )?;

        Ok(config)
    }

    /// Parse data type
    fn parse_data_type(&mut self) -> SklResult<DataType> {
        match &self.advance() {
            Token::Float32 => Ok(DataType::Float32),
            Token::Float64 => Ok(DataType::Float64),
            Token::Int32 => Ok(DataType::Int32),
            Token::Int64 => Ok(DataType::Int64),
            Token::Bool => Ok(DataType::Boolean),
            Token::String => Ok(DataType::String),
            Token::Array => {
                self.consume(Token::LeftAngle, "Expected '<' after 'Array'")?;
                let inner_type = self.parse_data_type()?;
                self.consume(Token::RightAngle, "Expected '>' after array type")?;
                Ok(DataType::Array(Box::new(inner_type)))
            }
            Token::Matrix => {
                self.consume(Token::LeftAngle, "Expected '<' after 'Matrix'")?;
                let inner_type = self.parse_data_type()?;
                self.consume(Token::RightAngle, "Expected '>' after matrix type")?;
                Ok(DataType::Matrix(Box::new(inner_type)))
            }
            Token::Identifier(name) => Ok(DataType::Custom(name.clone())),
            _ => Err(SklearsError::InvalidInput("Expected data type".to_string())),
        }
    }

    /// Parse parameter value
    fn parse_parameter_value(&mut self) -> SklResult<ParameterValue> {
        match &self.advance() {
            Token::NumberLiteral(n) => {
                if n.fract() == 0.0 {
                    Ok(ParameterValue::Int(*n as i64))
                } else {
                    Ok(ParameterValue::Float(*n))
                }
            }
            Token::BooleanLiteral(b) => Ok(ParameterValue::Bool(*b)),
            Token::StringLiteral(s) => Ok(ParameterValue::String(s.clone())),
            _ => Err(SklearsError::InvalidInput(
                "Expected parameter value".to_string(),
            )),
        }
    }

    /// Consume expected token
    fn consume(&mut self, expected: Token, message: &str) -> SklResult<Token> {
        if self.check(&expected) {
            Ok(self.advance())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "{}, got {:?}",
                message,
                self.peek()
            )))
        }
    }

    /// Check if current token matches expected
    fn check(&self, token_type: &Token) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(&self.peek()) == std::mem::discriminant(token_type)
        }
    }

    /// Advance to next token
    fn advance(&mut self) -> Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    /// Get current token
    fn peek(&self) -> Token {
        if let Some(token) = self.tokens.get(self.current) {
            token.clone()
        } else {
            Token::Eof
        }
    }

    /// Get previous token
    fn previous(&self) -> Token {
        if self.current > 0 {
            if let Some(token) = self.tokens.get(self.current - 1) {
                token.clone()
            } else {
                Token::Eof
            }
        } else {
            Token::Eof
        }
    }

    /// Check if at end of tokens
    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }
}

impl std::fmt::Display for DslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DSL Error at line {}, column {}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for DslError {}

impl Default for PipelineDSL {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DslLexer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DslParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Abstract Syntax Tree node for DSL parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AstNode {
    /// Pipeline node
    Pipeline {
        name: String,
        metadata: PipelineMetadata,
        children: Vec<AstNode>,
    },
    /// Step node
    Step {
        name: String,
        algorithm: String,
        parameters: Vec<AstNode>,
    },
    /// Connection node
    Connection {
        from: String,
        to: String,
        port_mapping: Vec<(String, String)>,
    },
    /// Parameter node
    Parameter { name: String, value: ParameterValue },
    /// Input definition node
    Input { name: String, data_type: DataType },
    /// Output definition node
    Output { name: String, data_type: DataType },
    /// Configuration node
    Config { key: String, value: String },
}

/// Pipeline metadata for AST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetadata {
    /// Pipeline version
    pub version: String,
    /// Pipeline description
    pub description: Option<String>,
    /// Pipeline author
    pub author: Option<String>,
}

/// Auto completer for DSL editing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoCompleter {
    /// Available keywords
    pub keywords: Vec<String>,
    /// Available functions
    pub functions: Vec<String>,
    /// Available components
    pub components: Vec<String>,
    /// Context-sensitive suggestions
    pub context_suggestions: BTreeMap<String, Vec<String>>,
}

/// DSL configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DslConfig {
    /// Enable syntax highlighting
    pub syntax_highlighting: bool,
    /// Enable auto completion
    pub auto_completion: bool,
    /// Enable real-time validation
    pub real_time_validation: bool,
    /// Indentation size
    pub indent_size: usize,
    /// Maximum line length
    pub max_line_length: usize,
    /// Comment style
    pub comment_style: CommentStyle,
}

/// Comment styles for DSL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommentStyle {
    /// Single line comments with //
    SingleLine,
    /// Multi-line comments with /* */
    MultiLine,
    /// Both styles supported
    Both,
}

/// Lexical analysis error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LexError {
    /// Unexpected character
    UnexpectedCharacter(char, usize, usize),
    /// Unterminated string
    UnterminatedString(usize, usize),
    /// Invalid number format
    InvalidNumber(String, usize, usize),
    /// Invalid escape sequence
    InvalidEscape(String, usize, usize),
    /// EOF reached unexpectedly
    UnexpectedEof(usize, usize),
}

/// Parse error types
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum ParseError {
    /// Unexpected token
    #[error("Unexpected token '{0}' at line {1}, column {2}")]
    UnexpectedToken(String, usize, usize),
    /// Missing token
    #[error("Missing token '{0}' at line {1}, column {2}")]
    MissingToken(String, usize, usize),
    /// Invalid syntax
    #[error("Invalid syntax: {0} at line {1}, column {2}")]
    InvalidSyntax(String, usize, usize),
    /// Semantic error
    #[error("Semantic error: {0} at line {1}, column {2}")]
    SemanticError(String, usize, usize),
    /// Unknown identifier
    #[error("Unknown identifier '{0}' at line {1}, column {2}")]
    UnknownIdentifier(String, usize, usize),
    /// Type mismatch
    #[error("Type mismatch: expected {1}, found {0} at line {2}, column {3}")]
    TypeMismatch(String, String, usize, usize),
}

/// Type alias for parse results
pub type ParseResult<T> = Result<T, ParseError>;

/// Semantic analyzer for DSL validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalyzer {
    /// Symbol table
    pub symbol_table: SymbolTable,
    /// Type checker
    pub type_checker: TypeChecker,
    /// Validation rules
    pub rules: Vec<SemanticRule>,
    /// Error collector
    pub errors: Vec<ParseError>,
}

/// Semantic validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule severity
    pub severity: RuleSeverity,
    /// Rule checker function name
    pub checker: String,
}

/// Rule severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleSeverity {
    /// Error - compilation fails
    Error,
    /// Warning - compilation succeeds with warning
    Warning,
    /// Info - informational message
    Info,
}

/// Symbol table for identifier tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTable {
    /// Defined symbols
    pub symbols: BTreeMap<String, Symbol>,
    /// Scope stack
    pub scopes: Vec<Scope>,
    /// Current scope level
    pub current_scope: usize,
}

/// Symbol definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    /// Symbol name
    pub name: String,
    /// Symbol type
    pub symbol_type: SymbolType,
    /// Data type
    pub data_type: DataType,
    /// Scope level
    pub scope: usize,
    /// Line number where defined
    pub line: usize,
    /// Column number where defined
    pub column: usize,
}

/// Symbol types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolType {
    /// Variable symbol
    Variable,
    /// Function symbol
    Function,
    /// Type symbol
    Type,
    /// Constant symbol
    Constant,
    /// Step symbol
    Step,
    /// Parameter symbol
    Parameter,
}

/// Scope definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    /// Scope name
    pub name: String,
    /// Parent scope
    pub parent: Option<usize>,
    /// Symbols in this scope
    pub symbols: Vec<String>,
}

/// Type alias for syntax highlighter
pub type SyntaxHighlighter = SyntaxHighlight;

/// Token types for lexical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenType {
    /// Keyword tokens
    Keyword(String),
    /// Identifier tokens
    Identifier(String),
    /// Number literal tokens
    Number(f64),
    /// String literal tokens
    String(String),
    /// Boolean literal tokens
    Boolean(bool),
    /// Operator tokens
    Operator(String),
    /// Punctuation tokens
    Punctuation(char),
    /// Comment tokens
    Comment(String),
    /// Whitespace tokens
    Whitespace(String),
    /// End of file token
    Eof,
}

/// Type checker for semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeChecker {
    /// Type rules
    pub rules: Vec<TypeRule>,
    /// Known types
    pub types: BTreeMap<String, TypeInfo>,
    /// Type coercion rules
    pub coercion_rules: Vec<CoercionRule>,
}

/// Type checking rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRule {
    /// Rule name
    pub name: String,
    /// Source type
    pub source_type: DataType,
    /// Target type
    pub target_type: DataType,
    /// Rule checker
    pub checker: String,
}

/// Type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type name
    pub name: String,
    /// Base type
    pub base_type: DataType,
    /// Type constraints
    pub constraints: Vec<String>,
    /// Type metadata
    pub metadata: BTreeMap<String, String>,
}

/// Type coercion rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoercionRule {
    /// From type
    pub from_type: DataType,
    /// To type
    pub to_type: DataType,
    /// Coercion cost
    pub cost: u32,
    /// Coercion function
    pub coercion_fn: String,
}

impl Default for DslConfig {
    fn default() -> Self {
        Self {
            syntax_highlighting: true,
            auto_completion: true,
            real_time_validation: true,
            indent_size: 4,
            max_line_length: 100,
            comment_style: CommentStyle::Both,
        }
    }
}

impl AutoCompleter {
    /// Create a new auto completer with default suggestions
    #[must_use]
    pub fn new() -> Self {
        let mut completer = Self {
            keywords: vec![
                "pipeline".to_string(),
                "step".to_string(),
                "connect".to_string(),
                "input".to_string(),
                "output".to_string(),
                "execute".to_string(),
                "version".to_string(),
            ],
            functions: vec![
                "transform".to_string(),
                "fit".to_string(),
                "predict".to_string(),
                "evaluate".to_string(),
            ],
            components: vec![
                "StandardScaler".to_string(),
                "LinearRegression".to_string(),
                "RandomForest".to_string(),
                "SVM".to_string(),
            ],
            context_suggestions: BTreeMap::new(),
        };

        // Add context-specific suggestions
        completer
            .context_suggestions
            .insert("step".to_string(), completer.components.clone());

        completer
    }

    /// Get suggestions for a given context
    #[must_use]
    pub fn get_suggestions(&self, context: &str, prefix: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Add keyword suggestions
        for keyword in &self.keywords {
            if keyword.starts_with(prefix) {
                suggestions.push(keyword.clone());
            }
        }

        // Add context-specific suggestions
        if let Some(context_suggestions) = self.context_suggestions.get(context) {
            for suggestion in context_suggestions {
                if suggestion.starts_with(prefix) {
                    suggestions.push(suggestion.clone());
                }
            }
        }

        suggestions.sort();
        suggestions.dedup();
        suggestions
    }
}

impl Default for AutoCompleter {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolTable {
    /// Create a new symbol table
    #[must_use]
    pub fn new() -> Self {
        Self {
            symbols: BTreeMap::new(),
            scopes: vec![Scope {
                name: "global".to_string(),
                parent: None,
                symbols: Vec::new(),
            }],
            current_scope: 0,
        }
    }

    /// Add a symbol to the current scope
    pub fn add_symbol(&mut self, symbol: Symbol) {
        self.symbols.insert(symbol.name.clone(), symbol.clone());
        if let Some(scope) = self.scopes.get_mut(self.current_scope) {
            scope.symbols.push(symbol.name);
        }
    }

    /// Look up a symbol
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeChecker {
    /// Create a new type checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            types: BTreeMap::new(),
            coercion_rules: Vec::new(),
        }
    }

    /// Check if two types are compatible
    #[must_use]
    pub fn are_compatible(&self, type1: &DataType, type2: &DataType) -> bool {
        type1 == type2 || self.can_coerce(type1, type2)
    }

    /// Check if type can be coerced
    #[must_use]
    pub fn can_coerce(&self, from: &DataType, to: &DataType) -> bool {
        self.coercion_rules
            .iter()
            .any(|rule| rule.from_type == *from && rule.to_type == *to)
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_basic_tokens() {
        let mut lexer = DslLexer::new();
        let tokens = lexer.tokenize("pipeline { }").unwrap();

        assert_eq!(tokens[0], Token::Pipeline);
        assert_eq!(tokens[1], Token::LeftBrace);
        assert_eq!(tokens[2], Token::RightBrace);
        assert_eq!(tokens[3], Token::Eof);
    }

    #[test]
    fn test_lexer_string_literal() {
        let mut lexer = DslLexer::new();
        let tokens = lexer.tokenize("\"hello world\"").unwrap();

        if let Token::StringLiteral(s) = &tokens[0] {
            assert_eq!(s, "hello world");
        } else {
            panic!("Expected string literal");
        }
    }

    #[test]
    fn test_lexer_number_literal() {
        let mut lexer = DslLexer::new();
        let tokens = lexer.tokenize("42.5").unwrap();

        if let Token::NumberLiteral(n) = &tokens[0] {
            assert_eq!(*n, 42.5);
        } else {
            panic!("Expected number literal");
        }
    }

    #[test]
    fn test_parser_simple_pipeline() {
        let mut dsl = PipelineDSL::new();
        let input = r#"
            pipeline "Test Pipeline" {
                version "1.0.0"
                step scaler: StandardScaler {
                    with_mean: true
                }
            }
        "#;

        let workflow = dsl.parse(input).unwrap();
        assert_eq!(workflow.metadata.name, "Test Pipeline");
        assert_eq!(workflow.metadata.version, "1.0.0");
        assert_eq!(workflow.steps.len(), 1);
        assert_eq!(workflow.steps[0].algorithm, "StandardScaler");
    }

    #[test]
    fn test_dsl_generation() {
        let mut workflow = WorkflowDefinition::default();
        workflow.metadata.name = "Test Workflow".to_string();
        workflow.metadata.version = "1.0.0".to_string();

        let step = StepDefinition::new("scaler", StepType::Transformer, "StandardScaler")
            .with_parameter("with_mean", ParameterValue::Bool(true));
        workflow.steps.push(step);

        let dsl = PipelineDSL::new();
        let generated = dsl.generate(&workflow);

        assert!(generated.contains("pipeline \"Test Workflow\""));
        assert!(generated.contains("version \"1.0.0\""));
        assert!(generated.contains("step scaler: StandardScaler"));
        assert!(generated.contains("with_mean: true"));
    }

    #[test]
    fn test_syntax_validation() {
        let mut dsl = PipelineDSL::new();

        // Valid syntax
        let valid_input = r#"
            pipeline "Valid" {
                version "1.0.0"
            }
        "#;
        let errors = dsl.validate_syntax(valid_input).unwrap();
        assert!(errors.is_empty());

        // Invalid syntax
        let invalid_input = r#"
            pipeline "Invalid" {
                version // missing value
            }
        "#;
        let errors = dsl.validate_syntax(invalid_input).unwrap();
        assert!(!errors.is_empty());
    }
}
