//! R integration utilities
//!
//! This module provides utilities for R integration including data exchange,
//! R script execution, statistical function bindings, and package management.

use std::collections::HashMap;
use std::fmt::Write;
use std::fs;
use std::process::Command;

/// R integration utilities
pub struct RIntegration {
    #[allow(dead_code)]
    r_home: Option<String>,
    #[allow(dead_code)]
    library_paths: Vec<String>,
    loaded_packages: Vec<String>,
    workspace_variables: HashMap<String, RValue>,
}

/// R value types
#[derive(Debug, Clone)]
pub enum RValue {
    Null,
    Logical(bool),
    Integer(i32),
    Double(f64),
    Character(String),
    IntegerVector(Vec<i32>),
    DoubleVector(Vec<f64>),
    CharacterVector(Vec<String>),
    LogicalVector(Vec<bool>),
    Matrix {
        data: Vec<f64>,
        nrows: usize,
        ncols: usize,
    },
    DataFrame {
        columns: HashMap<String, RValue>,
        nrows: usize,
    },
    List(HashMap<String, RValue>),
}

/// R data frame representation
#[derive(Debug, Clone)]
pub struct RDataFrame {
    pub columns: HashMap<String, RValue>,
    pub nrows: usize,
    pub column_names: Vec<String>,
}

/// R matrix representation
#[derive(Debug, Clone)]
pub struct RMatrix {
    pub data: Vec<f64>,
    pub nrows: usize,
    pub ncols: usize,
    pub row_names: Option<Vec<String>>,
    pub col_names: Option<Vec<String>>,
}

/// R script builder
pub struct RScriptBuilder {
    script_lines: Vec<String>,
    variables: HashMap<String, RValue>,
    packages: Vec<String>,
}

/// R package manager
pub struct RPackageManager {
    installed_packages: Vec<String>,
    #[allow(dead_code)]
    available_packages: HashMap<String, String>, // name -> version
    cran_mirror: String,
}

/// R statistical functions
pub struct RStatisticalFunctions;

impl RIntegration {
    /// Create new R integration instance
    pub fn new() -> Result<Self, RError> {
        let r_home = Self::detect_r_installation()?;

        Ok(Self {
            r_home: Some(r_home),
            library_paths: vec![],
            loaded_packages: ["base", "stats", "utils", "graphics", "grDevices"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            workspace_variables: HashMap::new(),
        })
    }

    /// Detect R installation
    fn detect_r_installation() -> Result<String, RError> {
        // Try to find R executable
        let output = Command::new("R")
            .args(["--slave", "--vanilla", "-e", "cat(R.home())"])
            .output()
            .map_err(|_| RError::RNotFound)?;

        if output.status.success() {
            String::from_utf8(output.stdout).map_err(|_| RError::InvalidOutput)
        } else {
            Err(RError::RNotFound)
        }
    }

    /// Execute R script
    pub fn execute_script(&mut self, script: &str) -> Result<String, RError> {
        // Write script to temporary file
        let temp_file = "/tmp/sklears_r_script.R";
        fs::write(temp_file, script).map_err(|e| RError::IoError(e.to_string()))?;

        // Execute R script
        let output = Command::new("Rscript")
            .arg(temp_file)
            .output()
            .map_err(|e| RError::ExecutionError(e.to_string()))?;

        // Clean up
        let _ = fs::remove_file(temp_file);

        if output.status.success() {
            String::from_utf8(output.stdout).map_err(|_| RError::InvalidOutput)
        } else {
            let error_msg =
                String::from_utf8(output.stderr).unwrap_or_else(|_| "Unknown R error".to_string());
            Err(RError::RScriptError(error_msg))
        }
    }

    /// Load R package
    pub fn load_package(&mut self, package_name: &str) -> Result<(), RError> {
        let script = format!("library({package_name})");
        self.execute_script(&script)?;
        self.loaded_packages.push(package_name.to_string());
        Ok(())
    }

    /// Convert Rust array to R vector
    pub fn array_to_r_vector(&self, data: &[f64]) -> RValue {
        RValue::DoubleVector(data.to_vec())
    }

    /// Convert Rust matrix to R matrix
    pub fn matrix_to_r_matrix(&self, data: &[f64], nrows: usize, ncols: usize) -> RValue {
        RValue::Matrix {
            data: data.to_vec(),
            nrows,
            ncols,
        }
    }

    /// Convert R value to Rust array
    pub fn r_vector_to_array(&self, r_value: &RValue) -> Result<Vec<f64>, RError> {
        match r_value {
            RValue::DoubleVector(vec) => Ok(vec.clone()),
            RValue::IntegerVector(vec) => Ok(vec.iter().map(|&x| x as f64).collect()),
            _ => Err(RError::TypeMismatch),
        }
    }

    /// Create R data frame from columns
    pub fn create_dataframe(&self, columns: HashMap<String, RValue>) -> Result<RDataFrame, RError> {
        // Validate all columns have same length
        let mut nrows = 0;
        let mut column_names = Vec::new();

        for (name, value) in &columns {
            let length = match value {
                RValue::IntegerVector(v) => v.len(),
                RValue::DoubleVector(v) => v.len(),
                RValue::CharacterVector(v) => v.len(),
                RValue::LogicalVector(v) => v.len(),
                _ => return Err(RError::InvalidDataFrame),
            };

            if nrows == 0 {
                nrows = length;
            } else if nrows != length {
                return Err(RError::InvalidDataFrame);
            }

            column_names.push(name.clone());
        }

        Ok(RDataFrame {
            columns,
            nrows,
            column_names,
        })
    }

    /// Execute R statistical function
    pub fn call_r_function(
        &mut self,
        function_name: &str,
        args: &[RValue],
    ) -> Result<RValue, RError> {
        let mut script = String::new();

        // Convert arguments to R syntax
        for (i, arg) in args.iter().enumerate() {
            let var_name = format!("arg{i}");
            let r_code = self.r_value_to_r_code(arg)?;
            writeln!(script, "{var_name} <- {r_code}")
                .map_err(|e| RError::ScriptGenerationError(e.to_string()))?;
        }

        // Call function
        let arg_names: Vec<String> = (0..args.len()).map(|i| format!("arg{i}")).collect();
        writeln!(
            script,
            "result <- {}({})",
            function_name,
            arg_names.join(", ")
        )
        .map_err(|e| RError::ScriptGenerationError(e.to_string()))?;

        // Output result
        writeln!(script, "cat(paste(result, collapse=','))")
            .map_err(|e| RError::ScriptGenerationError(e.to_string()))?;

        let output = self.execute_script(&script)?;
        self.parse_r_output(&output)
    }

    /// Convert R value to R code
    fn r_value_to_r_code(&self, value: &RValue) -> Result<String, RError> {
        match value {
            RValue::Null => Ok("NULL".to_string()),
            RValue::Logical(b) => Ok(if *b { "TRUE" } else { "FALSE" }.to_string()),
            RValue::Integer(i) => Ok(format!("{i}L")),
            RValue::Double(d) => Ok(d.to_string()),
            RValue::Character(s) => Ok(format!("\"{}\"", s.replace("\"", "\\\""))),
            RValue::IntegerVector(vec) => Ok(format!(
                "c({})",
                vec.iter()
                    .map(|x| format!("{x}L"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            RValue::DoubleVector(vec) => Ok(format!(
                "c({})",
                vec.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            RValue::CharacterVector(vec) => Ok(format!(
                "c({})",
                vec.iter()
                    .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            RValue::LogicalVector(vec) => Ok(format!(
                "c({})",
                vec.iter()
                    .map(|b| if *b { "TRUE" } else { "FALSE" })
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            RValue::Matrix { data, nrows, ncols } => Ok(format!(
                "matrix(c({}), nrow={}, ncol={})",
                data.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                nrows,
                ncols
            )),
            _ => Err(RError::UnsupportedType),
        }
    }

    /// Parse R output
    fn parse_r_output(&self, output: &str) -> Result<RValue, RError> {
        let trimmed = output.trim();

        if trimmed.is_empty() {
            return Ok(RValue::Null);
        }

        // Try to parse as comma-separated values
        if trimmed.contains(',') {
            let values: Result<Vec<f64>, _> = trimmed
                .split(',')
                .map(|s| s.trim().parse::<f64>())
                .collect();

            if let Ok(vec) = values {
                return Ok(RValue::DoubleVector(vec));
            }
        }

        // Try to parse as single value
        if let Ok(value) = trimmed.parse::<f64>() {
            return Ok(RValue::Double(value));
        }

        if let Ok(value) = trimmed.parse::<i32>() {
            return Ok(RValue::Integer(value));
        }

        if trimmed == "TRUE" {
            return Ok(RValue::Logical(true));
        }

        if trimmed == "FALSE" {
            return Ok(RValue::Logical(false));
        }

        // Default to character
        Ok(RValue::Character(trimmed.to_string()))
    }

    /// Get loaded packages
    pub fn get_loaded_packages(&self) -> &[String] {
        &self.loaded_packages
    }

    /// Check if package is available
    pub fn is_package_available(&mut self, package_name: &str) -> Result<bool, RError> {
        let script = format!("cat(is.element('{package_name}', installed.packages()[,1]))");
        let output = self.execute_script(&script)?;
        Ok(output.trim() == "TRUE")
    }

    /// Install R package
    pub fn install_package(&mut self, package_name: &str) -> Result<(), RError> {
        let script =
            format!("install.packages('{package_name}', repos='https://cran.r-project.org')");
        self.execute_script(&script)?;
        Ok(())
    }

    /// Save workspace variable
    pub fn save_variable(&mut self, name: &str, value: RValue) {
        self.workspace_variables.insert(name.to_string(), value);
    }

    /// Get workspace variable
    pub fn get_variable(&self, name: &str) -> Option<&RValue> {
        self.workspace_variables.get(name)
    }

    /// Clear workspace
    pub fn clear_workspace(&mut self) {
        self.workspace_variables.clear();
    }
}

impl RScriptBuilder {
    /// Create new R script builder
    pub fn new() -> Self {
        Self {
            script_lines: Vec::new(),
            variables: HashMap::new(),
            packages: Vec::new(),
        }
    }

    /// Add package requirement
    pub fn require_package(&mut self, package: &str) -> &mut Self {
        self.packages.push(package.to_string());
        self
    }

    /// Add variable assignment
    pub fn assign_variable(&mut self, name: &str, value: RValue) -> &mut Self {
        self.variables.insert(name.to_string(), value);
        self
    }

    /// Add R code line
    pub fn add_line(&mut self, line: &str) -> &mut Self {
        self.script_lines.push(line.to_string());
        self
    }

    /// Add comment
    pub fn add_comment(&mut self, comment: &str) -> &mut Self {
        self.script_lines.push(format!("# {comment}"));
        self
    }

    /// Build the complete R script
    pub fn build(&self) -> Result<String, RError> {
        let mut script = String::new();

        // Add package loading
        for package in &self.packages {
            writeln!(script, "library({package})")
                .map_err(|e| RError::ScriptGenerationError(e.to_string()))?;
        }

        if !self.packages.is_empty() {
            writeln!(script).map_err(|e| RError::ScriptGenerationError(e.to_string()))?;
        }

        // Add variable assignments
        for (name, value) in &self.variables {
            let r_code = self.r_value_to_r_code(value)?;
            writeln!(script, "{name} <- {r_code}")
                .map_err(|e| RError::ScriptGenerationError(e.to_string()))?;
        }

        if !self.variables.is_empty() {
            writeln!(script).map_err(|e| RError::ScriptGenerationError(e.to_string()))?;
        }

        // Add script lines
        for line in &self.script_lines {
            writeln!(script, "{line}").map_err(|e| RError::ScriptGenerationError(e.to_string()))?;
        }

        Ok(script)
    }

    /// Convert R value to R code (helper method)
    fn r_value_to_r_code(&self, value: &RValue) -> Result<String, RError> {
        match value {
            RValue::Null => Ok("NULL".to_string()),
            RValue::Logical(b) => Ok(if *b { "TRUE" } else { "FALSE" }.to_string()),
            RValue::Integer(i) => Ok(format!("{i}L")),
            RValue::Double(d) => Ok(d.to_string()),
            RValue::Character(s) => Ok(format!("\"{}\"", s.replace("\"", "\\\""))),
            RValue::DoubleVector(vec) => Ok(format!(
                "c({})",
                vec.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            _ => Err(RError::UnsupportedType),
        }
    }
}

impl RPackageManager {
    /// Create new R package manager
    pub fn new() -> Self {
        Self {
            installed_packages: Vec::new(),
            available_packages: HashMap::new(),
            cran_mirror: "https://cran.r-project.org".to_string(),
        }
    }

    /// Refresh package information
    pub fn refresh(&mut self) -> Result<(), RError> {
        // Get installed packages
        let script = "cat(paste(installed.packages()[,1], collapse=','))";
        let output = Command::new("Rscript")
            .args(["-e", script])
            .output()
            .map_err(|e| RError::ExecutionError(e.to_string()))?;

        if output.status.success() {
            let packages_str =
                String::from_utf8(output.stdout).map_err(|_| RError::InvalidOutput)?;

            self.installed_packages = packages_str
                .trim()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        Ok(())
    }

    /// Check if package is installed
    pub fn is_installed(&self, package_name: &str) -> bool {
        self.installed_packages.contains(&package_name.to_string())
    }

    /// Install package
    pub fn install(&mut self, package_name: &str) -> Result<(), RError> {
        let script = format!(
            "install.packages('{}', repos='{}')",
            package_name, self.cran_mirror
        );

        let output = Command::new("Rscript")
            .args(["-e", &script])
            .output()
            .map_err(|e| RError::ExecutionError(e.to_string()))?;

        if output.status.success() {
            self.installed_packages.push(package_name.to_string());
            Ok(())
        } else {
            let error_msg = String::from_utf8(output.stderr)
                .unwrap_or_else(|_| "Package installation failed".to_string());
            Err(RError::PackageInstallationError(error_msg))
        }
    }

    /// Remove package
    pub fn remove(&mut self, package_name: &str) -> Result<(), RError> {
        let script = format!("remove.packages('{package_name}')");

        let output = Command::new("Rscript")
            .args(["-e", &script])
            .output()
            .map_err(|e| RError::ExecutionError(e.to_string()))?;

        if output.status.success() {
            self.installed_packages.retain(|p| p != package_name);
            Ok(())
        } else {
            let error_msg = String::from_utf8(output.stderr)
                .unwrap_or_else(|_| "Package removal failed".to_string());
            Err(RError::PackageRemovalError(error_msg))
        }
    }

    /// Get installed packages
    pub fn get_installed_packages(&self) -> &[String] {
        &self.installed_packages
    }

    /// Set CRAN mirror
    pub fn set_cran_mirror(&mut self, mirror_url: &str) {
        self.cran_mirror = mirror_url.to_string();
    }
}

impl RStatisticalFunctions {
    /// Compute mean using R
    pub fn mean(r: &mut RIntegration, data: &[f64]) -> Result<f64, RError> {
        let r_vector = r.array_to_r_vector(data);
        let result = r.call_r_function("mean", &[r_vector])?;

        match result {
            RValue::Double(mean_val) => Ok(mean_val),
            _ => Err(RError::TypeMismatch),
        }
    }

    /// Compute standard deviation using R
    pub fn sd(r: &mut RIntegration, data: &[f64]) -> Result<f64, RError> {
        let r_vector = r.array_to_r_vector(data);
        let result = r.call_r_function("sd", &[r_vector])?;

        match result {
            RValue::Double(sd_val) => Ok(sd_val),
            _ => Err(RError::TypeMismatch),
        }
    }

    /// Perform t-test using R
    pub fn t_test(r: &mut RIntegration, x: &[f64], y: &[f64]) -> Result<RValue, RError> {
        r.load_package("stats")?;

        let x_vector = r.array_to_r_vector(x);
        let y_vector = r.array_to_r_vector(y);

        r.call_r_function("t.test", &[x_vector, y_vector])
    }

    /// Perform linear regression using R
    pub fn lm(r: &mut RIntegration, x: &[f64], y: &[f64]) -> Result<RValue, RError> {
        r.load_package("stats")?;

        // Create data frame
        let mut columns = HashMap::new();
        columns.insert("x".to_string(), r.array_to_r_vector(x));
        columns.insert("y".to_string(), r.array_to_r_vector(y));

        let script = "
data <- data.frame(x=arg0, y=arg1)
model <- lm(y ~ x, data=data)
coefficients <- coef(model)
cat(paste(coefficients, collapse=','))
";

        let mut script_with_args = String::new();
        writeln!(
            script_with_args,
            "arg0 <- {}",
            r.r_value_to_r_code(&columns["x"])?
        )
        .unwrap();
        writeln!(
            script_with_args,
            "arg1 <- {}",
            r.r_value_to_r_code(&columns["y"])?
        )
        .unwrap();
        script_with_args.push_str(script);

        let output = r.execute_script(&script_with_args)?;
        r.parse_r_output(&output)
    }

    /// Compute correlation using R
    pub fn cor(r: &mut RIntegration, x: &[f64], y: &[f64]) -> Result<f64, RError> {
        let x_vector = r.array_to_r_vector(x);
        let y_vector = r.array_to_r_vector(y);
        let result = r.call_r_function("cor", &[x_vector, y_vector])?;

        match result {
            RValue::Double(cor_val) => Ok(cor_val),
            _ => Err(RError::TypeMismatch),
        }
    }

    /// Perform ANOVA using R
    pub fn anova(r: &mut RIntegration, groups: &[Vec<f64>]) -> Result<RValue, RError> {
        r.load_package("stats")?;

        // Prepare data for ANOVA
        let mut script = String::new();
        let mut all_values = Vec::new();
        let mut group_labels = Vec::new();

        for (i, group) in groups.iter().enumerate() {
            for &value in group {
                all_values.push(value);
                let group_num = i + 1;
                group_labels.push(format!("Group{group_num}"));
            }
        }

        writeln!(
            script,
            "values <- c({})",
            all_values
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(
            script,
            "groups <- factor(c({}))",
            group_labels
                .iter()
                .map(|s| format!("\"{s}\""))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(script, "result <- aov(values ~ groups)").unwrap();
        writeln!(script, "summary_result <- summary(result)").unwrap();
        writeln!(script, "cat('ANOVA completed')").unwrap();

        let output = r.execute_script(&script)?;
        Ok(RValue::Character(output))
    }

    /// Generate R plots
    pub fn plot(r: &mut RIntegration, x: &[f64], y: &[f64], filename: &str) -> Result<(), RError> {
        let script = format!(
            "
x <- c({})
y <- c({})
png('{}')
plot(x, y, main='Scatter Plot', xlab='X', ylab='Y')
dev.off()
",
            x.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            y.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            filename
        );

        r.execute_script(&script)?;
        Ok(())
    }
}

/// R integration errors
#[derive(Debug, thiserror::Error)]
pub enum RError {
    #[error("R installation not found")]
    RNotFound,
    #[error("Invalid R output")]
    InvalidOutput,
    #[error("I/O error: {0}")]
    IoError(String),
    #[error("R script execution error: {0}")]
    ExecutionError(String),
    #[error("R script error: {0}")]
    RScriptError(String),
    #[error("Type mismatch")]
    TypeMismatch,
    #[error("Invalid data frame")]
    InvalidDataFrame,
    #[error("Unsupported type")]
    UnsupportedType,
    #[error("Script generation error: {0}")]
    ScriptGenerationError(String),
    #[error("Package installation error: {0}")]
    PackageInstallationError(String),
    #[error("Package removal error: {0}")]
    PackageRemovalError(String),
}

impl Default for RIntegration {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            r_home: None,
            library_paths: Vec::new(),
            loaded_packages: Vec::new(),
            workspace_variables: HashMap::new(),
        })
    }
}

impl Default for RScriptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RPackageManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_script_builder() {
        let mut builder = RScriptBuilder::new();

        builder
            .require_package("stats")
            .assign_variable("x", RValue::DoubleVector(vec![1.0, 2.0, 3.0]))
            .add_comment("Calculate mean")
            .add_line("mean_x <- mean(x)")
            .add_line("cat(mean_x)");

        let script = builder.build().unwrap();
        assert!(script.contains("library(stats)"));
        assert!(script.contains("x <- c(1, 2, 3)"));
        assert!(script.contains("# Calculate mean"));
        assert!(script.contains("mean_x <- mean(x)"));
    }

    #[test]
    fn test_r_value_conversions() {
        let r = RIntegration::default();

        // Test array to R vector
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let r_vector = r.array_to_r_vector(&data);

        match r_vector {
            RValue::DoubleVector(ref vec) => assert_eq!(vec, &data),
            _ => panic!("Expected DoubleVector"),
        }

        // Test R vector to array
        let array = r.r_vector_to_array(&r_vector).unwrap();
        assert_eq!(array, data);
    }

    #[test]
    fn test_dataframe_creation() {
        let r = RIntegration::default();

        let mut columns = HashMap::new();
        columns.insert("x".to_string(), RValue::DoubleVector(vec![1.0, 2.0, 3.0]));
        columns.insert("y".to_string(), RValue::DoubleVector(vec![4.0, 5.0, 6.0]));

        let df = r.create_dataframe(columns).unwrap();
        assert_eq!(df.nrows, 3);
        assert_eq!(df.column_names.len(), 2);
    }

    #[test]
    fn test_r_code_generation() {
        let r = RIntegration::default();

        // Test various R value types
        assert_eq!(
            r.r_value_to_r_code(&RValue::Double(std::f64::consts::PI))
                .unwrap(),
            format!("{}", std::f64::consts::PI)
        );
        assert_eq!(r.r_value_to_r_code(&RValue::Integer(42)).unwrap(), "42L");
        assert_eq!(r.r_value_to_r_code(&RValue::Logical(true)).unwrap(), "TRUE");
        assert_eq!(
            r.r_value_to_r_code(&RValue::Character("test".to_string()))
                .unwrap(),
            "\"test\""
        );

        let vec_result = r
            .r_value_to_r_code(&RValue::DoubleVector(vec![1.0, 2.0, 3.0]))
            .unwrap();
        assert_eq!(vec_result, "c(1, 2, 3)");

        let matrix_result = r
            .r_value_to_r_code(&RValue::Matrix {
                data: vec![1.0, 2.0, 3.0, 4.0],
                nrows: 2,
                ncols: 2,
            })
            .unwrap();
        assert_eq!(matrix_result, "matrix(c(1, 2, 3, 4), nrow=2, ncol=2)");
    }

    #[test]
    fn test_package_manager() {
        let mut manager = RPackageManager::new();

        // Test initial state
        assert_eq!(manager.get_installed_packages().len(), 0);
        assert!(!manager.is_installed("ggplot2"));

        // Test adding packages manually (for testing)
        manager.installed_packages.push("base".to_string());
        manager.installed_packages.push("stats".to_string());

        assert!(manager.is_installed("base"));
        assert!(manager.is_installed("stats"));
        assert!(!manager.is_installed("ggplot2"));
    }

    #[test]
    fn test_workspace_variables() {
        let mut r = RIntegration::default();

        let value = RValue::Double(42.0);
        r.save_variable("test_var", value.clone());

        let retrieved = r.get_variable("test_var");
        assert!(retrieved.is_some());

        match retrieved.unwrap() {
            RValue::Double(val) => assert_eq!(*val, 42.0),
            _ => panic!("Expected Double value"),
        }

        r.clear_workspace();
        assert!(r.get_variable("test_var").is_none());
    }

    #[test]
    fn test_output_parsing() {
        let r = RIntegration::default();

        // Test parsing different output types
        let result = r.parse_r_output("42.5").unwrap();
        match result {
            RValue::Double(val) => assert_eq!(val, 42.5),
            _ => panic!("Expected Double"),
        }

        let result = r.parse_r_output("1,2,3,4").unwrap();
        match result {
            RValue::DoubleVector(vec) => assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]),
            _ => panic!("Expected DoubleVector"),
        }

        let result = r.parse_r_output("TRUE").unwrap();
        match result {
            RValue::Logical(val) => assert!(val),
            _ => panic!("Expected Logical"),
        }

        let result = r.parse_r_output("test string").unwrap();
        match result {
            RValue::Character(val) => assert_eq!(val, "test string"),
            _ => panic!("Expected Character"),
        }
    }

    #[test]
    fn test_matrix_operations() {
        let r = RIntegration::default();

        let matrix = r.matrix_to_r_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        match matrix {
            RValue::Matrix { data, nrows, ncols } => {
                assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
                assert_eq!(nrows, 2);
                assert_eq!(ncols, 2);
            }
            _ => panic!("Expected Matrix"),
        }
    }

    // Note: The following tests would require R to be installed and available
    // They are commented out but show how integration testing would work

    /*
    #[test]
    fn test_r_integration_with_real_r() {
        let mut r = RIntegration::new().unwrap();

        // Test simple calculation
        let result = r.execute_script("cat(2 + 2)").unwrap();
        assert_eq!(result.trim(), "4");

        // Test statistical function
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean_result = RStatisticalFunctions::mean(&mut r, &data).unwrap();
        assert_eq!(mean_result, 3.0);
    }

    #[test]
    fn test_package_operations() {
        let mut r = RIntegration::new().unwrap();

        // Test loading base package
        assert!(r.load_package("stats").is_ok());
        assert!(r.get_loaded_packages().contains(&"stats".to_string()));

        // Test package availability check
        let is_available = r.is_package_available("stats").unwrap();
        assert!(is_available);
    }
    */
}
