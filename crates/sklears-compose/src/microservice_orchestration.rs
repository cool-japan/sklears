use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc, watch};
use uuid::Uuid;

use crate::api_gateway::{MLPipelineGateway, ServiceEndpoint, GatewayError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroserviceWorkflow {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub steps: Vec<WorkflowStep>,
    pub timeout: Duration,
    pub retry_policy: RetryPolicy,
    pub rollback_strategy: RollbackStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub id: String,
    pub service_name: String,
    pub operation: String,
    pub input_mapping: HashMap<String, String>,
    pub output_mapping: HashMap<String, String>,
    pub dependencies: Vec<String>,
    pub timeout: Option<Duration>,
    pub retry_count: u32,
    pub compensation_action: Option<CompensationAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompensationAction {
    pub service_name: String,
    pub operation: String,
    pub input_mapping: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryPolicy {
    None,
    Fixed { attempts: u32, delay: Duration },
    Exponential { attempts: u32, initial_delay: Duration, multiplier: f64 },
    Linear { attempts: u32, initial_delay: Duration, increment: Duration },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackStrategy {
    None,
    Compensating,
    Snapshot,
    EventSourcing,
}

#[derive(Debug, Clone)]
pub struct WorkflowExecution {
    pub workflow_id: Uuid,
    pub execution_id: Uuid,
    pub status: WorkflowStatus,
    pub started_at: Instant,
    pub completed_at: Option<Instant>,
    pub step_results: HashMap<String, StepResult>,
    pub compensation_log: Vec<CompensationRecord>,
    pub context: WorkflowContext,
}

#[derive(Debug, Clone)]
pub enum WorkflowStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Compensating,
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_id: String,
    pub status: StepStatus,
    pub started_at: Instant,
    pub completed_at: Option<Instant>,
    pub output: Option<String>,
    pub error: Option<String>,
    pub retry_count: u32,
}

#[derive(Debug, Clone)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
    Compensated,
}

#[derive(Debug, Clone)]
pub struct CompensationRecord {
    pub step_id: String,
    pub action: CompensationAction,
    pub executed_at: Instant,
    pub result: Result<String, String>,
}

#[derive(Debug, Clone)]
pub struct WorkflowContext {
    pub variables: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, thiserror::Error)]
pub enum OrchestrationError {
    #[error("Workflow not found: {workflow_id}")]
    WorkflowNotFound { workflow_id: Uuid },
    
    #[error("Workflow execution not found: {execution_id}")]
    ExecutionNotFound { execution_id: Uuid },
    
    #[error("Circular dependency detected in workflow: {workflow_id}")]
    CircularDependency { workflow_id: Uuid },
    
    #[error("Step dependency not satisfied: {step_id} depends on {dependency}")]
    DependencyNotSatisfied { step_id: String, dependency: String },
    
    #[error("Workflow execution timeout: {execution_id}")]
    ExecutionTimeout { execution_id: Uuid },
    
    #[error("Step execution failed: {step_id} - {error}")]
    StepExecutionFailed { step_id: String, error: String },
    
    #[error("Compensation failed: {step_id} - {error}")]
    CompensationFailed { step_id: String, error: String },
    
    #[error("Gateway error: {0}")]
    Gateway(#[from] GatewayError),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub struct MicroserviceOrchestrator {
    gateway: Arc<MLPipelineGateway>,
    workflows: Arc<RwLock<HashMap<Uuid, MicroserviceWorkflow>>>,
    executions: Arc<RwLock<HashMap<Uuid, WorkflowExecution>>>,
    execution_queue: Arc<RwLock<VecDeque<Uuid>>>,
    event_bus: Arc<EventBus>,
    metrics: Arc<RwLock<OrchestrationMetrics>>,
}

#[derive(Debug, Default)]
pub struct OrchestrationMetrics {
    pub total_workflows: u64,
    pub active_executions: u64,
    pub completed_executions: u64,
    pub failed_executions: u64,
    pub average_execution_time: Duration,
    pub compensation_rate: f64,
}

pub struct EventBus {
    workflow_events: watch::Sender<WorkflowEvent>,
    step_events: mpsc::UnboundedSender<StepEvent>,
}

#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    WorkflowStarted { execution_id: Uuid, workflow_id: Uuid },
    WorkflowCompleted { execution_id: Uuid, duration: Duration },
    WorkflowFailed { execution_id: Uuid, error: String },
    WorkflowCancelled { execution_id: Uuid },
}

#[derive(Debug, Clone)]
pub enum StepEvent {
    StepStarted { execution_id: Uuid, step_id: String },
    StepCompleted { execution_id: Uuid, step_id: String, output: String },
    StepFailed { execution_id: Uuid, step_id: String, error: String },
    StepRetrying { execution_id: Uuid, step_id: String, attempt: u32 },
}

impl MicroserviceOrchestrator {
    pub fn new(gateway: Arc<MLPipelineGateway>) -> Self {
        let (workflow_tx, _workflow_rx) = watch::channel(WorkflowEvent::WorkflowStarted {
            execution_id: Uuid::new_v4(),
            workflow_id: Uuid::new_v4(),
        });
        let (step_tx, _step_rx) = mpsc::unbounded_channel();
        
        Self {
            gateway,
            workflows: Arc::new(RwLock::new(HashMap::new())),
            executions: Arc::new(RwLock::new(HashMap::new())),
            execution_queue: Arc::new(RwLock::new(VecDeque::new())),
            event_bus: Arc::new(EventBus {
                workflow_events: workflow_tx,
                step_events: step_tx,
            }),
            metrics: Arc::new(RwLock::new(OrchestrationMetrics::default())),
        }
    }

    pub async fn register_workflow(&self, workflow: MicroserviceWorkflow) -> Result<(), OrchestrationError> {
        self.validate_workflow(&workflow)?;
        
        let mut workflows = self.workflows.write().await;
        workflows.insert(workflow.id, workflow);
        
        let mut metrics = self.metrics.write().await;
        metrics.total_workflows += 1;
        
        Ok(())
    }

    pub async fn execute_workflow(
        &self,
        workflow_id: Uuid,
        initial_context: WorkflowContext,
    ) -> Result<Uuid, OrchestrationError> {
        let workflow = {
            let workflows = self.workflows.read().await;
            workflows.get(&workflow_id)
                .ok_or(OrchestrationError::WorkflowNotFound { workflow_id })?
                .clone()
        };

        let execution_id = Uuid::new_v4();
        let execution = WorkflowExecution {
            workflow_id,
            execution_id,
            status: WorkflowStatus::Pending,
            started_at: Instant::now(),
            completed_at: None,
            step_results: HashMap::new(),
            compensation_log: Vec::new(),
            context: initial_context,
        };

        {
            let mut executions = self.executions.write().await;
            executions.insert(execution_id, execution);
        }

        {
            let mut queue = self.execution_queue.write().await;
            queue.push_back(execution_id);
        }

        let mut metrics = self.metrics.write().await;
        metrics.active_executions += 1;

        let _ = self.event_bus.workflow_events.send(WorkflowEvent::WorkflowStarted {
            execution_id,
            workflow_id,
        });

        tokio::spawn(self.clone().process_execution_queue());
        
        Ok(execution_id)
    }

    async fn process_execution_queue(self: Arc<Self>) {
        while let Some(execution_id) = {
            let mut queue = self.execution_queue.write().await;
            queue.pop_front()
        } {
            if let Err(e) = self.execute_workflow_steps(execution_id).await {
                log::error!("Workflow execution failed: {:?}", e);
                self.handle_workflow_failure(execution_id, e).await;
            }
        }
    }

    async fn execute_workflow_steps(&self, execution_id: Uuid) -> Result<(), OrchestrationError> {
        let workflow = {
            let executions = self.executions.read().await;
            let execution = executions.get(&execution_id)
                .ok_or(OrchestrationError::ExecutionNotFound { execution_id })?;
            
            let workflows = self.workflows.read().await;
            workflows.get(&execution.workflow_id)
                .ok_or(OrchestrationError::WorkflowNotFound { 
                    workflow_id: execution.workflow_id 
                })?
                .clone()
        };

        {
            let mut executions = self.executions.write().await;
            if let Some(execution) = executions.get_mut(&execution_id) {
                execution.status = WorkflowStatus::Running;
            }
        }

        let execution_order = self.calculate_execution_order(&workflow)?;
        
        for step_id in execution_order {
            if let Some(step) = workflow.steps.iter().find(|s| s.id == step_id) {
                self.execute_step(execution_id, step.clone()).await?;
            }
        }

        self.complete_workflow_execution(execution_id).await;
        Ok(())
    }

    async fn execute_step(&self, execution_id: Uuid, step: WorkflowStep) -> Result<(), OrchestrationError> {
        let step_result = StepResult {
            step_id: step.id.clone(),
            status: StepStatus::Running,
            started_at: Instant::now(),
            completed_at: None,
            output: None,
            error: None,
            retry_count: 0,
        };

        {
            let mut executions = self.executions.write().await;
            if let Some(execution) = executions.get_mut(&execution_id) {
                execution.step_results.insert(step.id.clone(), step_result);
            }
        }

        let _ = self.event_bus.step_events.send(StepEvent::StepStarted {
            execution_id,
            step_id: step.id.clone(),
        });

        let input_data = self.prepare_step_input(execution_id, &step).await?;
        
        let mut attempt = 0;
        loop {
            attempt += 1;
            
            match self.call_service(&step.service_name, &step.operation, &input_data).await {
                Ok(output) => {
                    let processed_output = self.process_step_output(&step, &output).await?;
                    self.complete_step_execution(execution_id, &step.id, processed_output).await;
                    break;
                }
                Err(e) => {
                    if attempt >= step.retry_count {
                        self.fail_step_execution(execution_id, &step.id, format!("{:?}", e)).await;
                        return Err(OrchestrationError::StepExecutionFailed {
                            step_id: step.id,
                            error: format!("{:?}", e),
                        });
                    } else {
                        let _ = self.event_bus.step_events.send(StepEvent::StepRetrying {
                            execution_id,
                            step_id: step.id.clone(),
                            attempt,
                        });
                        
                        tokio::time::sleep(Duration::from_millis(1000 * attempt as u64)).await;
                    }
                }
            }
        }

        Ok(())
    }

    async fn call_service(&self, service_name: &str, operation: &str, input_data: &str) -> Result<String, GatewayError> {
        let method = if operation.contains("get") || operation.contains("fetch") {
            "GET"
        } else {
            "POST"
        };
        
        let path = format!("/{}", operation);
        let headers = HashMap::new();
        
        self.gateway.route_request(method, &path, input_data, headers).await
    }

    async fn prepare_step_input(&self, execution_id: Uuid, step: &WorkflowStep) -> Result<String, OrchestrationError> {
        let executions = self.executions.read().await;
        let execution = executions.get(&execution_id)
            .ok_or(OrchestrationError::ExecutionNotFound { execution_id })?;

        let mut input_data = HashMap::new();
        
        for (output_key, input_key) in &step.input_mapping {
            if let Some(value) = execution.context.variables.get(output_key) {
                input_data.insert(input_key.clone(), value.clone());
            } else {
                for (_, step_result) in &execution.step_results {
                    if step_result.status == StepStatus::Completed {
                        if let Some(output) = &step_result.output {
                            if let Ok(output_json) = serde_json::from_str::<HashMap<String, serde_json::Value>>(output) {
                                if let Some(value) = output_json.get(output_key) {
                                    input_data.insert(input_key.clone(), value.to_string());
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(serde_json::to_string(&input_data)?)
    }

    async fn process_step_output(&self, step: &WorkflowStep, output: &str) -> Result<String, OrchestrationError> {
        if step.output_mapping.is_empty() {
            return Ok(output.to_string());
        }

        let output_json: HashMap<String, serde_json::Value> = serde_json::from_str(output)?;
        let mut processed_output = HashMap::new();
        
        for (source_key, target_key) in &step.output_mapping {
            if let Some(value) = output_json.get(source_key) {
                processed_output.insert(target_key.clone(), value.clone());
            }
        }

        Ok(serde_json::to_string(&processed_output)?)
    }

    async fn complete_step_execution(&self, execution_id: Uuid, step_id: &str, output: String) {
        let mut executions = self.executions.write().await;
        if let Some(execution) = executions.get_mut(&execution_id) {
            if let Some(step_result) = execution.step_results.get_mut(step_id) {
                step_result.status = StepStatus::Completed;
                step_result.completed_at = Some(Instant::now());
                step_result.output = Some(output.clone());
            }
        }

        let _ = self.event_bus.step_events.send(StepEvent::StepCompleted {
            execution_id,
            step_id: step_id.to_string(),
            output,
        });
    }

    async fn fail_step_execution(&self, execution_id: Uuid, step_id: &str, error: String) {
        let mut executions = self.executions.write().await;
        if let Some(execution) = executions.get_mut(&execution_id) {
            if let Some(step_result) = execution.step_results.get_mut(step_id) {
                step_result.status = StepStatus::Failed;
                step_result.completed_at = Some(Instant::now());
                step_result.error = Some(error.clone());
            }
        }

        let _ = self.event_bus.step_events.send(StepEvent::StepFailed {
            execution_id,
            step_id: step_id.to_string(),
            error,
        });
    }

    async fn complete_workflow_execution(&self, execution_id: Uuid) {
        let duration = {
            let mut executions = self.executions.write().await;
            if let Some(execution) = executions.get_mut(&execution_id) {
                execution.status = WorkflowStatus::Completed;
                execution.completed_at = Some(Instant::now());
                execution.started_at.elapsed()
            } else {
                Duration::from_secs(0)
            }
        };

        let mut metrics = self.metrics.write().await;
        metrics.active_executions = metrics.active_executions.saturating_sub(1);
        metrics.completed_executions += 1;
        
        let total_completed = metrics.completed_executions as f64;
        let current_avg = metrics.average_execution_time.as_secs_f64();
        let new_avg = (current_avg * (total_completed - 1.0) + duration.as_secs_f64()) / total_completed;
        metrics.average_execution_time = Duration::from_secs_f64(new_avg);

        let _ = self.event_bus.workflow_events.send(WorkflowEvent::WorkflowCompleted {
            execution_id,
            duration,
        });
    }

    async fn handle_workflow_failure(&self, execution_id: Uuid, error: OrchestrationError) {
        {
            let mut executions = self.executions.write().await;
            if let Some(execution) = executions.get_mut(&execution_id) {
                execution.status = WorkflowStatus::Failed;
                execution.completed_at = Some(Instant::now());
            }
        }

        let mut metrics = self.metrics.write().await;
        metrics.active_executions = metrics.active_executions.saturating_sub(1);
        metrics.failed_executions += 1;

        let _ = self.event_bus.workflow_events.send(WorkflowEvent::WorkflowFailed {
            execution_id,
            error: format!("{:?}", error),
        });

        if let Err(compensation_error) = self.execute_compensation(execution_id).await {
            log::error!("Compensation failed: {:?}", compensation_error);
        }
    }

    async fn execute_compensation(&self, execution_id: Uuid) -> Result<(), OrchestrationError> {
        let workflow_and_execution = {
            let executions = self.executions.read().await;
            let execution = executions.get(&execution_id)
                .ok_or(OrchestrationError::ExecutionNotFound { execution_id })?;
            
            let workflows = self.workflows.read().await;
            let workflow = workflows.get(&execution.workflow_id)
                .ok_or(OrchestrationError::WorkflowNotFound { 
                    workflow_id: execution.workflow_id 
                })?;
                
            (workflow.clone(), execution.clone())
        };

        let (workflow, execution) = workflow_and_execution;

        {
            let mut executions = self.executions.write().await;
            if let Some(exec) = executions.get_mut(&execution_id) {
                exec.status = WorkflowStatus::Compensating;
            }
        }

        for step in workflow.steps.iter().rev() {
            if let Some(step_result) = execution.step_results.get(&step.id) {
                if step_result.status == StepStatus::Completed {
                    if let Some(compensation) = &step.compensation_action {
                        match self.execute_compensation_action(execution_id, step, compensation).await {
                            Ok(result) => {
                                let record = CompensationRecord {
                                    step_id: step.id.clone(),
                                    action: compensation.clone(),
                                    executed_at: Instant::now(),
                                    result: Ok(result),
                                };
                                
                                let mut executions = self.executions.write().await;
                                if let Some(exec) = executions.get_mut(&execution_id) {
                                    exec.compensation_log.push(record);
                                }
                            }
                            Err(e) => {
                                let record = CompensationRecord {
                                    step_id: step.id.clone(),
                                    action: compensation.clone(),
                                    executed_at: Instant::now(),
                                    result: Err(format!("{:?}", e)),
                                };
                                
                                let mut executions = self.executions.write().await;
                                if let Some(exec) = executions.get_mut(&execution_id) {
                                    exec.compensation_log.push(record);
                                }
                                
                                return Err(OrchestrationError::CompensationFailed {
                                    step_id: step.id.clone(),
                                    error: format!("{:?}", e),
                                });
                            }
                        }
                    }
                }
            }
        }

        let mut metrics = self.metrics.write().await;
        let total_failed = metrics.failed_executions as f64;
        let compensated = metrics.failed_executions as f64;
        metrics.compensation_rate = compensated / total_failed;

        Ok(())
    }

    async fn execute_compensation_action(
        &self,
        _execution_id: Uuid,
        _step: &WorkflowStep,
        compensation: &CompensationAction,
    ) -> Result<String, GatewayError> {
        let input_data = serde_json::to_string(&compensation.input_mapping).unwrap_or_default();
        self.call_service(&compensation.service_name, &compensation.operation, &input_data).await
    }

    fn calculate_execution_order(&self, workflow: &MicroserviceWorkflow) -> Result<Vec<String>, OrchestrationError> {
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut temp_visited = std::collections::HashSet::new();
        
        for step in &workflow.steps {
            if !visited.contains(&step.id) {
                self.dfs_visit(
                    &step.id,
                    &workflow.steps,
                    &mut visited,
                    &mut temp_visited,
                    &mut order,
                )?;
            }
        }
        
        order.reverse();
        Ok(order)
    }

    fn dfs_visit(
        &self,
        step_id: &str,
        steps: &[WorkflowStep],
        visited: &mut std::collections::HashSet<String>,
        temp_visited: &mut std::collections::HashSet<String>,
        order: &mut Vec<String>,
    ) -> Result<(), OrchestrationError> {
        if temp_visited.contains(step_id) {
            return Err(OrchestrationError::CircularDependency {
                workflow_id: Uuid::new_v4(),
            });
        }
        
        if visited.contains(step_id) {
            return Ok(());
        }
        
        temp_visited.insert(step_id.to_string());
        
        if let Some(step) = steps.iter().find(|s| s.id == step_id) {
            for dependency in &step.dependencies {
                self.dfs_visit(dependency, steps, visited, temp_visited, order)?;
            }
        }
        
        temp_visited.remove(step_id);
        visited.insert(step_id.to_string());
        order.push(step_id.to_string());
        
        Ok(())
    }

    fn validate_workflow(&self, workflow: &MicroserviceWorkflow) -> Result<(), OrchestrationError> {
        let step_ids: std::collections::HashSet<_> = workflow.steps.iter().map(|s| &s.id).collect();
        
        for step in &workflow.steps {
            for dependency in &step.dependencies {
                if !step_ids.contains(dependency) {
                    return Err(OrchestrationError::DependencyNotSatisfied {
                        step_id: step.id.clone(),
                        dependency: dependency.clone(),
                    });
                }
            }
        }
        
        Ok(())
    }

    pub async fn get_execution_status(&self, execution_id: Uuid) -> Option<WorkflowExecution> {
        let executions = self.executions.read().await;
        executions.get(&execution_id).cloned()
    }

    pub async fn cancel_execution(&self, execution_id: Uuid) -> Result<(), OrchestrationError> {
        let mut executions = self.executions.write().await;
        if let Some(execution) = executions.get_mut(&execution_id) {
            execution.status = WorkflowStatus::Cancelled;
            execution.completed_at = Some(Instant::now());
            
            let _ = self.event_bus.workflow_events.send(WorkflowEvent::WorkflowCancelled {
                execution_id,
            });
            
            Ok(())
        } else {
            Err(OrchestrationError::ExecutionNotFound { execution_id })
        }
    }

    pub async fn get_metrics(&self) -> OrchestrationMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }
}

impl Clone for MicroserviceOrchestrator {
    fn clone(&self) -> Self {
        Self {
            gateway: self.gateway.clone(),
            workflows: self.workflows.clone(),
            executions: self.executions.clone(),
            execution_queue: self.execution_queue.clone(),
            event_bus: self.event_bus.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_gateway::MLPipelineGateway;

    #[tokio::test]
    async fn test_workflow_registration() {
        let gateway = Arc::new(MLPipelineGateway::new());
        let orchestrator = MicroserviceOrchestrator::new(gateway);
        
        let workflow = MicroserviceWorkflow {
            id: Uuid::new_v4(),
            name: "test-workflow".to_string(),
            description: "Test workflow".to_string(),
            steps: vec![
                WorkflowStep {
                    id: "step1".to_string(),
                    service_name: "service1".to_string(),
                    operation: "operation1".to_string(),
                    input_mapping: HashMap::new(),
                    output_mapping: HashMap::new(),
                    dependencies: vec![],
                    timeout: Some(Duration::from_secs(30)),
                    retry_count: 3,
                    compensation_action: None,
                },
            ],
            timeout: Duration::from_secs(300),
            retry_policy: RetryPolicy::Fixed {
                attempts: 3,
                delay: Duration::from_secs(1),
            },
            rollback_strategy: RollbackStrategy::Compensating,
        };
        
        assert!(orchestrator.register_workflow(workflow).await.is_ok());
    }

    #[tokio::test]
    async fn test_execution_order_calculation() {
        let gateway = Arc::new(MLPipelineGateway::new());
        let orchestrator = MicroserviceOrchestrator::new(gateway);
        
        let workflow = MicroserviceWorkflow {
            id: Uuid::new_v4(),
            name: "dependency-test".to_string(),
            description: "Test workflow with dependencies".to_string(),
            steps: vec![
                WorkflowStep {
                    id: "step1".to_string(),
                    service_name: "service1".to_string(),
                    operation: "op1".to_string(),
                    input_mapping: HashMap::new(),
                    output_mapping: HashMap::new(),
                    dependencies: vec![],
                    timeout: None,
                    retry_count: 0,
                    compensation_action: None,
                },
                WorkflowStep {
                    id: "step2".to_string(),
                    service_name: "service2".to_string(),
                    operation: "op2".to_string(),
                    input_mapping: HashMap::new(),
                    output_mapping: HashMap::new(),
                    dependencies: vec!["step1".to_string()],
                    timeout: None,
                    retry_count: 0,
                    compensation_action: None,
                },
            ],
            timeout: Duration::from_secs(300),
            retry_policy: RetryPolicy::None,
            rollback_strategy: RollbackStrategy::None,
        };
        
        let order = orchestrator.calculate_execution_order(&workflow).unwrap();
        assert_eq!(order, vec!["step1", "step2"]);
    }

    #[tokio::test]
    async fn test_circular_dependency_detection() {
        let gateway = Arc::new(MLPipelineGateway::new());
        let orchestrator = MicroserviceOrchestrator::new(gateway);
        
        let workflow = MicroserviceWorkflow {
            id: Uuid::new_v4(),
            name: "circular-test".to_string(),
            description: "Test workflow with circular dependencies".to_string(),
            steps: vec![
                WorkflowStep {
                    id: "step1".to_string(),
                    service_name: "service1".to_string(),
                    operation: "op1".to_string(),
                    input_mapping: HashMap::new(),
                    output_mapping: HashMap::new(),
                    dependencies: vec!["step2".to_string()],
                    timeout: None,
                    retry_count: 0,
                    compensation_action: None,
                },
                WorkflowStep {
                    id: "step2".to_string(),
                    service_name: "service2".to_string(),
                    operation: "op2".to_string(),
                    input_mapping: HashMap::new(),
                    output_mapping: HashMap::new(),
                    dependencies: vec!["step1".to_string()],
                    timeout: None,
                    retry_count: 0,
                    compensation_action: None,
                },
            ],
            timeout: Duration::from_secs(300),
            retry_policy: RetryPolicy::None,
            rollback_strategy: RollbackStrategy::None,
        };
        
        let result = orchestrator.calculate_execution_order(&workflow);
        assert!(matches!(result, Err(OrchestrationError::CircularDependency { .. })));
    }

    #[tokio::test]
    async fn test_workflow_validation() {
        let gateway = Arc::new(MLPipelineGateway::new());
        let orchestrator = MicroserviceOrchestrator::new(gateway);
        
        let invalid_workflow = MicroserviceWorkflow {
            id: Uuid::new_v4(),
            name: "invalid-workflow".to_string(),
            description: "Invalid workflow".to_string(),
            steps: vec![
                WorkflowStep {
                    id: "step1".to_string(),
                    service_name: "service1".to_string(),
                    operation: "op1".to_string(),
                    input_mapping: HashMap::new(),
                    output_mapping: HashMap::new(),
                    dependencies: vec!["nonexistent_step".to_string()],
                    timeout: None,
                    retry_count: 0,
                    compensation_action: None,
                },
            ],
            timeout: Duration::from_secs(300),
            retry_policy: RetryPolicy::None,
            rollback_strategy: RollbackStrategy::None,
        };
        
        let result = orchestrator.validate_workflow(&invalid_workflow);
        assert!(matches!(result, Err(OrchestrationError::DependencyNotSatisfied { .. })));
    }
}