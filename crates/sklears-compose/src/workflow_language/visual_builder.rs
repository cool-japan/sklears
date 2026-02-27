//! Visual Pipeline Builder
//!
//! This module provides visual pipeline building capabilities for creating machine learning
//! workflows through a graphical interface, including drag-and-drop component assembly,
//! visual connection management, and interactive workflow construction.

use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{BTreeMap, HashMap};

use super::workflow_definitions::{Connection, ParameterValue, StepDefinition, WorkflowDefinition};

/// Visual pipeline builder for creating workflows graphically
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualPipelineBuilder {
    /// Current workflow being built
    pub workflow: WorkflowDefinition,
    /// Component positioning information
    pub component_positions: HashMap<String, Position>,
    /// Canvas configuration
    pub canvas_config: CanvasConfig,
    /// Validation state
    pub validation_state: ValidationState,
    /// Undo/redo history
    pub history: Vec<WorkflowSnapshot>,
    /// Current history index
    pub history_index: usize,
}

/// Position of a component on the visual canvas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Width of the component
    pub width: f64,
    /// Height of the component
    pub height: f64,
}

/// Canvas configuration for the visual builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasConfig {
    /// Canvas width
    pub width: f64,
    /// Canvas height
    pub height: f64,
    /// Grid size for snapping
    pub grid_size: f64,
    /// Zoom level
    pub zoom: f64,
    /// Pan offset X
    pub pan_x: f64,
    /// Pan offset Y
    pub pan_y: f64,
    /// Enable grid snapping
    pub snap_to_grid: bool,
    /// Show grid lines
    pub show_grid: bool,
}

/// Validation state for the workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationState {
    /// Whether the workflow is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Last validation timestamp
    pub last_validated: String,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,
    /// Error message
    pub message: String,
    /// Step ID where error occurred (if applicable)
    pub step_id: Option<String>,
    /// Connection ID where error occurred (if applicable)
    pub connection_id: Option<String>,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning type
    pub warning_type: ValidationWarningType,
    /// Warning message
    pub message: String,
    /// Step ID where warning occurred (if applicable)
    pub step_id: Option<String>,
}

/// Types of validation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationErrorType {
    /// Missing required input
    MissingInput,
    /// Type mismatch between connected steps
    TypeMismatch,
    /// Circular dependency detected
    CircularDependency,
    /// Disconnected component
    DisconnectedComponent,
    /// Invalid parameter value
    InvalidParameter,
    /// Missing required step
    MissingStep,
}

/// Types of validation warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationWarningType {
    /// Unused output
    UnusedOutput,
    /// Performance concern
    PerformanceConcern,
    /// Deprecated component
    DeprecatedComponent,
    /// Suboptimal configuration
    SuboptimalConfiguration,
}

/// Workflow snapshot for undo/redo functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSnapshot {
    /// Snapshot of the workflow state
    pub workflow: WorkflowDefinition,
    /// Snapshot of component positions
    pub positions: HashMap<String, Position>,
    /// Description of the action that created this snapshot
    pub action_description: String,
    /// Timestamp of the snapshot
    pub timestamp: String,
}

impl VisualPipelineBuilder {
    /// Create a new visual pipeline builder
    #[must_use]
    pub fn new() -> Self {
        let workflow = WorkflowDefinition::default();
        let initial_snapshot = WorkflowSnapshot {
            workflow: workflow.clone(),
            positions: HashMap::new(),
            action_description: "Initial state".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        Self {
            workflow,
            component_positions: HashMap::new(),
            canvas_config: CanvasConfig::default(),
            validation_state: ValidationState::new(),
            history: vec![initial_snapshot],
            history_index: 0,
        }
    }

    /// Add a step to the workflow
    pub fn add_step(&mut self, step: StepDefinition) -> SklResult<()> {
        // Check if step ID already exists
        if self.workflow.steps.iter().any(|s| s.id == step.id) {
            return Err(SklearsError::InvalidInput(format!(
                "Step with ID '{}' already exists",
                step.id
            )));
        }

        // Add step to workflow
        self.workflow.steps.push(step.clone());

        // Add default position
        let position = self.find_optimal_position(&step.id);
        self.component_positions.insert(step.id.clone(), position);

        // Create snapshot for undo/redo
        self.create_snapshot(&format!("Added step '{}'", step.id));

        // Validate workflow and update state
        let _ = self.validate();

        Ok(())
    }

    /// Remove a step from the workflow
    pub fn remove_step(&mut self, step_id: &str) -> SklResult<()> {
        // Remove step
        let initial_count = self.workflow.steps.len();
        self.workflow.steps.retain(|s| s.id != step_id);

        if self.workflow.steps.len() == initial_count {
            return Err(SklearsError::InvalidInput(format!(
                "Step '{step_id}' not found"
            )));
        }

        // Remove position
        self.component_positions.remove(step_id);

        // Remove connections involving this step
        self.workflow
            .connections
            .retain(|c| c.from_step != step_id && c.to_step != step_id);

        // Create snapshot
        self.create_snapshot(&format!("Removed step '{step_id}'"));

        // Validate workflow and update state
        let _ = self.validate();

        Ok(())
    }

    /// Add a connection between steps
    pub fn add_connection(&mut self, connection: Connection) -> SklResult<()> {
        // Validate that both steps exist
        let from_exists = self
            .workflow
            .steps
            .iter()
            .any(|s| s.id == connection.from_step);
        let to_exists = self
            .workflow
            .steps
            .iter()
            .any(|s| s.id == connection.to_step);

        if !from_exists {
            return Err(SklearsError::InvalidInput(format!(
                "Source step '{}' not found",
                connection.from_step
            )));
        }

        if !to_exists {
            return Err(SklearsError::InvalidInput(format!(
                "Target step '{}' not found",
                connection.to_step
            )));
        }

        // Check for duplicate connections
        let duplicate = self.workflow.connections.iter().any(|c| {
            c.from_step == connection.from_step
                && c.from_output == connection.from_output
                && c.to_step == connection.to_step
                && c.to_input == connection.to_input
        });

        if duplicate {
            return Err(SklearsError::InvalidInput(
                "Connection already exists".to_string(),
            ));
        }

        // Add connection
        self.workflow.connections.push(connection.clone());

        // Create snapshot
        self.create_snapshot(&format!(
            "Added connection from '{}' to '{}'",
            connection.from_step, connection.to_step
        ));

        // Validate workflow and update state
        let _ = self.validate();

        Ok(())
    }

    /// Remove a connection
    pub fn remove_connection(
        &mut self,
        from_step: &str,
        from_output: &str,
        to_step: &str,
        to_input: &str,
    ) -> SklResult<()> {
        let initial_count = self.workflow.connections.len();

        self.workflow.connections.retain(|c| {
            !(c.from_step == from_step
                && c.from_output == from_output
                && c.to_step == to_step
                && c.to_input == to_input)
        });

        if self.workflow.connections.len() == initial_count {
            return Err(SklearsError::InvalidInput(
                "Connection not found".to_string(),
            ));
        }

        // Create snapshot
        self.create_snapshot(&format!(
            "Removed connection from '{from_step}' to '{to_step}'"
        ));

        // Validate workflow and update state
        let _ = self.validate();

        Ok(())
    }

    /// Move a component to a new position
    pub fn move_component(&mut self, step_id: &str, position: Position) -> SklResult<()> {
        if !self.workflow.steps.iter().any(|s| s.id == step_id) {
            return Err(SklearsError::InvalidInput(format!(
                "Step '{step_id}' not found"
            )));
        }

        let final_position = if self.canvas_config.snap_to_grid {
            self.snap_to_grid(position)
        } else {
            position
        };

        self.component_positions
            .insert(step_id.to_string(), final_position);

        // Create snapshot for significant moves (optional - might be too frequent)
        // self.create_snapshot(&format!("Moved component '{}'", step_id));

        Ok(())
    }

    /// Update step parameters
    pub fn update_step_parameters(
        &mut self,
        step_id: &str,
        parameters: BTreeMap<String, ParameterValue>,
    ) -> SklResult<()> {
        let step = self
            .workflow
            .steps
            .iter_mut()
            .find(|s| s.id == step_id)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Step '{step_id}' not found")))?;

        step.parameters = parameters;

        // Create snapshot
        self.create_snapshot(&format!("Updated parameters for '{step_id}'"));

        // Validate workflow and update state
        let _ = self.validate();

        Ok(())
    }

    /// Validate the current workflow and update the validation state.
    ///
    /// This method performs a comprehensive validation pass, updating
    /// `self.validation_state` with the latest errors and warnings. It does
    /// **not** short-circuit the caller when issues are detected; instead,
    /// callers are expected to inspect the returned [`ValidationState`] and
    /// decide whether to proceed.
    #[must_use]
    pub fn validate(&mut self) -> &ValidationState {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for disconnected components
        for step in &self.workflow.steps {
            let has_input = self
                .workflow
                .connections
                .iter()
                .any(|c| c.to_step == step.id);
            let has_output = self
                .workflow
                .connections
                .iter()
                .any(|c| c.from_step == step.id);

            if !has_input && !has_output && self.workflow.steps.len() > 1 {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::DisconnectedComponent,
                    message: format!(
                        "Step '{}' is not connected to any other components",
                        step.id
                    ),
                    step_id: Some(step.id.clone()),
                    connection_id: None,
                });
            }
        }

        // Check for circular dependencies
        if self.has_circular_dependency() {
            errors.push(ValidationError {
                error_type: ValidationErrorType::CircularDependency,
                message: "Circular dependency detected in workflow".to_string(),
                step_id: None,
                connection_id: None,
            });
        }

        // Check for unused outputs
        for step in &self.workflow.steps {
            for output in &step.outputs {
                let is_used = self
                    .workflow
                    .connections
                    .iter()
                    .any(|c| c.from_step == step.id && c.from_output == *output);

                if !is_used {
                    warnings.push(ValidationWarning {
                        warning_type: ValidationWarningType::UnusedOutput,
                        message: format!("Output '{}' of step '{}' is not used", output, step.id),
                        step_id: Some(step.id.clone()),
                    });
                }
            }
        }

        self.validation_state = ValidationState {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            last_validated: chrono::Utc::now().to_rfc3339(),
        };

        &self.validation_state
    }

    /// Check for circular dependencies in the workflow
    fn has_circular_dependency(&self) -> bool {
        let mut visited = HashMap::new();
        let mut recursion_stack = HashMap::new();

        for step in &self.workflow.steps {
            if !visited.get(&step.id).unwrap_or(&false)
                && self.has_cycle_util(&step.id, &mut visited, &mut recursion_stack)
            {
                return true;
            }
        }

        false
    }

    /// Utility function for cycle detection
    fn has_cycle_util(
        &self,
        step_id: &str,
        visited: &mut HashMap<String, bool>,
        recursion_stack: &mut HashMap<String, bool>,
    ) -> bool {
        visited.insert(step_id.to_string(), true);
        recursion_stack.insert(step_id.to_string(), true);

        // Get all steps that depend on this step
        for connection in &self.workflow.connections {
            if connection.from_step == step_id {
                let next_step = &connection.to_step;

                if !visited.get(next_step).unwrap_or(&false) {
                    if self.has_cycle_util(next_step, visited, recursion_stack) {
                        return true;
                    }
                } else if *recursion_stack.get(next_step).unwrap_or(&false) {
                    return true;
                }
            }
        }

        recursion_stack.insert(step_id.to_string(), false);
        false
    }

    /// Find optimal position for a new component
    fn find_optimal_position(&self, _step_id: &str) -> Position {
        // Simple positioning logic - place components in a grid
        let num_components = self.component_positions.len();
        let grid_cols = 4;
        let component_width = 120.0;
        let component_height = 80.0;
        let spacing_x = 160.0;
        let spacing_y = 120.0;

        let col = num_components % grid_cols;
        let row = num_components / grid_cols;

        Position {
            x: 50.0 + col as f64 * spacing_x,
            y: 50.0 + row as f64 * spacing_y,
            width: component_width,
            height: component_height,
        }
    }

    /// Snap position to grid
    #[must_use]
    pub fn snap_to_grid(&self, position: Position) -> Position {
        let grid_size = self.canvas_config.grid_size;
        Position {
            x: (position.x / grid_size).round() * grid_size,
            y: (position.y / grid_size).round() * grid_size,
            width: position.width,
            height: position.height,
        }
    }

    /// Create a snapshot for undo/redo
    fn create_snapshot(&mut self, action_description: &str) {
        let snapshot = WorkflowSnapshot {
            workflow: self.workflow.clone(),
            positions: self.component_positions.clone(),
            action_description: action_description.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Remove any snapshots after current index (when creating new branch)
        self.history.truncate(self.history_index + 1);

        // Add new snapshot
        self.history.push(snapshot);
        self.history_index = self.history.len() - 1;

        // Limit history size
        const MAX_HISTORY_SIZE: usize = 50;
        if self.history.len() > MAX_HISTORY_SIZE {
            self.history.remove(0);
            self.history_index = self.history.len() - 1;
        }
    }

    /// Undo last action
    pub fn undo(&mut self) -> SklResult<()> {
        if self.history_index == 0 {
            return Err(SklearsError::InvalidInput("Nothing to undo".to_string()));
        }

        self.history_index -= 1;
        let snapshot = &self.history[self.history_index];

        self.workflow = snapshot.workflow.clone();
        self.component_positions = snapshot.positions.clone();

        Ok(())
    }

    /// Redo last undone action
    pub fn redo(&mut self) -> SklResult<()> {
        if self.history_index >= self.history.len() - 1 {
            return Err(SklearsError::InvalidInput("Nothing to redo".to_string()));
        }

        self.history_index += 1;
        let snapshot = &self.history[self.history_index];

        self.workflow = snapshot.workflow.clone();
        self.component_positions = snapshot.positions.clone();

        Ok(())
    }

    /// Get current workflow
    #[must_use]
    pub fn get_workflow(&self) -> &WorkflowDefinition {
        &self.workflow
    }

    /// Get component positions
    #[must_use]
    pub fn get_component_positions(&self) -> &HashMap<String, Position> {
        &self.component_positions
    }

    /// Get validation state
    #[must_use]
    pub fn get_validation_state(&self) -> &ValidationState {
        &self.validation_state
    }

    /// Clear the workflow
    pub fn clear(&mut self) {
        self.workflow = WorkflowDefinition::default();
        self.component_positions.clear();
        self.validation_state = ValidationState::new();

        // Create snapshot
        self.create_snapshot("Cleared workflow");
    }

    /// Set canvas configuration
    pub fn set_canvas_config(&mut self, config: CanvasConfig) {
        self.canvas_config = config;
    }

    /// Get canvas configuration
    #[must_use]
    pub fn get_canvas_config(&self) -> &CanvasConfig {
        &self.canvas_config
    }
}

impl Default for VisualPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CanvasConfig {
    fn default() -> Self {
        Self {
            width: 1200.0,
            height: 800.0,
            grid_size: 20.0,
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            snap_to_grid: true,
            show_grid: true,
        }
    }
}

impl ValidationState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            last_validated: chrono::Utc::now().to_rfc3339(),
        }
    }
}

impl Default for ValidationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for component position
pub type ComponentPosition = Position;

/// Canvas interaction state for drag-and-drop operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasInteraction {
    /// Current interaction mode
    pub mode: InteractionMode,
    /// Selected components
    pub selected_components: Vec<String>,
    /// Current drag state
    pub drag_state: Option<DragState>,
    /// Selection state
    pub selection_state: SelectionState,
}

/// Interaction modes for the canvas
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InteractionMode {
    /// Normal selection mode
    Select,
    /// Panning the canvas
    Pan,
    /// Zooming the canvas
    Zoom,
    /// Drawing connections
    Connect,
    /// Adding new components
    AddComponent,
}

/// Drag and drop state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragState {
    /// Component being dragged
    pub component_id: String,
    /// Start position of drag
    pub start_position: Position,
    /// Current position during drag
    pub current_position: Position,
    /// Offset from component center
    pub offset: Position,
}

/// Selection state management
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SelectionState {
    /// Currently selected components
    pub selected: Vec<String>,
    /// Selection box coordinates (if active)
    pub selection_box: Option<SelectionBox>,
    /// Multi-select mode enabled
    pub multi_select: bool,
}

/// Selection box for area selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionBox {
    /// Start corner of selection box
    pub start: Position,
    /// End corner of selection box
    pub end: Position,
}

/// Type alias for grid configuration
pub type GridConfig = CanvasConfig;

/// Viewport configuration for canvas display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportConfig {
    /// Viewport width
    pub width: f64,
    /// Viewport height
    pub height: f64,
    /// Pan offset X
    pub pan_x: f64,
    /// Pan offset Y
    pub pan_y: f64,
    /// Zoom level
    pub zoom: f64,
}

/// Zoom configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomConfig {
    /// Current zoom level
    pub level: f64,
    /// Minimum zoom level
    pub min_zoom: f64,
    /// Maximum zoom level
    pub max_zoom: f64,
    /// Zoom step size
    pub zoom_step: f64,
    /// Zoom center point
    pub center: Position,
}

/// Type alias for workflow history
pub type WorkflowHistory = Vec<WorkflowSnapshot>;

/// Undo/Redo manager for workflow operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoRedoManager {
    /// History of workflow snapshots
    pub history: WorkflowHistory,
    /// Current position in history
    pub current_index: usize,
    /// Maximum history size
    pub max_history_size: usize,
}

impl Default for CanvasInteraction {
    fn default() -> Self {
        Self {
            mode: InteractionMode::Select,
            selected_components: Vec::new(),
            drag_state: None,
            selection_state: SelectionState::default(),
        }
    }
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            width: 1200.0,
            height: 800.0,
            pan_x: 0.0,
            pan_y: 0.0,
            zoom: 1.0,
        }
    }
}

impl Default for ZoomConfig {
    fn default() -> Self {
        Self {
            level: 1.0,
            min_zoom: 0.1,
            max_zoom: 5.0,
            zoom_step: 0.1,
            center: Position {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
            },
        }
    }
}

impl UndoRedoManager {
    /// Create a new undo/redo manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            current_index: 0,
            max_history_size: 50,
        }
    }

    /// Add a new snapshot to history
    pub fn add_snapshot(&mut self, snapshot: WorkflowSnapshot) {
        // Remove any snapshots after current index
        self.history.truncate(self.current_index + 1);

        // Add new snapshot
        self.history.push(snapshot);
        self.current_index = self.history.len() - 1;

        // Limit history size
        if self.history.len() > self.max_history_size {
            self.history.remove(0);
            self.current_index = self.history.len() - 1;
        }
    }

    /// Check if undo is available
    #[must_use]
    pub fn can_undo(&self) -> bool {
        self.current_index > 0
    }

    /// Check if redo is available
    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.current_index < self.history.len() - 1
    }
}

impl Default for UndoRedoManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow_language::workflow_definitions::{DataType, StepType};

    #[test]
    fn test_visual_pipeline_builder_creation() {
        let builder = VisualPipelineBuilder::new();
        assert_eq!(builder.workflow.steps.len(), 0);
        assert_eq!(builder.component_positions.len(), 0);
        assert!(builder.validation_state.is_valid);
        assert_eq!(builder.history.len(), 1); // Initial snapshot
    }

    #[test]
    fn test_add_step() {
        let mut builder = VisualPipelineBuilder::new();

        let step = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");
        let result = builder.add_step(step);

        assert!(result.is_ok());
        assert_eq!(builder.workflow.steps.len(), 1);
        assert_eq!(builder.component_positions.len(), 1);
        assert!(builder.component_positions.contains_key("step1"));
    }

    #[test]
    fn test_add_duplicate_step() {
        let mut builder = VisualPipelineBuilder::new();

        let step1 = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");
        let step2 = StepDefinition::new("step1", StepType::Predictor, "LinearRegression");

        assert!(builder.add_step(step1).is_ok());
        assert!(builder.add_step(step2).is_err());
    }

    #[test]
    fn test_remove_step() {
        let mut builder = VisualPipelineBuilder::new();

        let step = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");
        builder.add_step(step).unwrap();

        assert_eq!(builder.workflow.steps.len(), 1);

        let result = builder.remove_step("step1");
        assert!(result.is_ok());
        assert_eq!(builder.workflow.steps.len(), 0);
        assert!(!builder.component_positions.contains_key("step1"));
    }

    #[test]
    fn test_add_connection() {
        let mut builder = VisualPipelineBuilder::new();

        let step1 = StepDefinition::new("step1", StepType::Transformer, "StandardScaler")
            .with_output("X_scaled");
        let step2 =
            StepDefinition::new("step2", StepType::Predictor, "LinearRegression").with_input("X");

        builder.add_step(step1).unwrap();
        builder.add_step(step2).unwrap();

        let connection = Connection::direct("step1", "X_scaled", "step2", "X");
        let result = builder.add_connection(connection);

        assert!(result.is_ok());
        assert_eq!(builder.workflow.connections.len(), 1);
    }

    #[test]
    fn test_undo_redo() {
        let mut builder = VisualPipelineBuilder::new();

        let step = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");
        builder.add_step(step).unwrap();

        assert_eq!(builder.workflow.steps.len(), 1);
        assert_eq!(builder.history_index, 1);

        // Undo
        builder.undo().unwrap();
        assert_eq!(builder.workflow.steps.len(), 0);
        assert_eq!(builder.history_index, 0);

        // Redo
        builder.redo().unwrap();
        assert_eq!(builder.workflow.steps.len(), 1);
        assert_eq!(builder.history_index, 1);
    }

    #[test]
    fn test_snap_to_grid() {
        let builder = VisualPipelineBuilder::new();
        let position = Position {
            x: 23.7,
            y: 47.3,
            width: 100.0,
            height: 80.0,
        };

        let snapped = builder.snap_to_grid(position);
        assert_eq!(snapped.x, 20.0);
        assert_eq!(snapped.y, 40.0);
    }

    #[test]
    fn test_validation_disconnected_component() {
        let mut builder = VisualPipelineBuilder::new();

        let step1 = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");
        let step2 = StepDefinition::new("step2", StepType::Predictor, "LinearRegression");

        builder.add_step(step1).unwrap();
        builder.add_step(step2).unwrap();

        // Both steps are disconnected, validation should fail
        assert!(!builder.validation_state.is_valid);
        assert!(!builder.validation_state.errors.is_empty());
    }
}
