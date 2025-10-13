//! Response Team Coordination System
//!
//! This module provides comprehensive response team coordination capabilities including
//! team mobilization, deployment tracking, task assignment, capacity management,
//! and multi-team coordination for emergency response operations.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

// Import types from sibling modules
use super::detection::{EmergencyEvent, EmergencyType, EmergencySeverity};

/// Response team coordination and deployment system
///
/// Manages response team mobilization, deployment tracking, task assignment,
/// and coordination between multiple teams during emergency response operations.
/// Provides comprehensive team management with skill-based routing and capacity planning.
#[derive(Debug)]
pub struct ResponseTeamCoordinator {
    /// Response teams registry
    response_teams: Arc<RwLock<HashMap<String, ResponseTeam>>>,
    /// Active team deployments
    active_deployments: Arc<RwLock<HashMap<String, TeamDeployment>>>,
    /// Team capacity tracking
    capacity_tracker: Arc<RwLock<TeamCapacityTracker>>,
    /// Task assignment registry
    task_assignments: Arc<RwLock<HashMap<String, TaskAssignment>>>,
    /// Coordination metrics
    coordination_metrics: Arc<RwLock<CoordinationMetrics>>,
    /// Deployment history
    deployment_history: Arc<RwLock<Vec<HistoricalDeployment>>>,
}

impl ResponseTeamCoordinator {
    pub fn new() -> Self {
        Self {
            response_teams: Arc::new(RwLock::new(HashMap::new())),
            active_deployments: Arc::new(RwLock::new(HashMap::new())),
            capacity_tracker: Arc::new(RwLock::new(TeamCapacityTracker::new())),
            task_assignments: Arc::new(RwLock::new(HashMap::new())),
            coordination_metrics: Arc::new(RwLock::new(CoordinationMetrics::new())),
            deployment_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn initialize(&self) -> SklResult<()> {
        self.setup_response_teams()?;
        self.initialize_capacity_tracking()?;
        Ok(())
    }

    /// Mobilize appropriate response teams based on emergency
    pub fn mobilize_teams(&self, event: &EmergencyEvent) -> SklResult<MobilizationResult> {
        let selected_teams = self.select_teams_for_emergency(event)?;
        let mut deployment_results = Vec::new();

        for team_id in selected_teams {
            let deployment = self.deploy_team(&team_id, event)?;
            deployment_results.push(deployment);
        }

        let mobilization_result = MobilizationResult {
            mobilization_id: uuid::Uuid::new_v4().to_string(),
            emergency_id: event.event_id.clone(),
            teams_mobilized: deployment_results.len(),
            deployment_ids: deployment_results.iter().map(|d| d.deployment_id.clone()).collect(),
            mobilization_time: Duration::from_minutes(15), // Estimated
            success: true,
            estimated_response_capacity: self.calculate_response_capacity(&deployment_results)?,
        };

        // Update coordination metrics
        {
            let mut metrics = self.coordination_metrics.write()
                .map_err(|_| SklearsError::Other("Failed to acquire metrics lock".into()))?;
            metrics.total_mobilizations += 1;
            metrics.teams_mobilized += deployment_results.len() as u64;
            metrics.last_mobilization = Some(SystemTime::now());
        }

        Ok(mobilization_result)
    }

    /// Deploy a specific team
    pub fn deploy_team(&self, team_id: &str, event: &EmergencyEvent) -> SklResult<TeamDeployment> {
        let team = {
            let teams = self.response_teams.read()
                .map_err(|_| SklearsError::Other("Failed to acquire teams lock".into()))?;
            teams.get(team_id).cloned()
                .ok_or_else(|| SklearsError::InvalidInput(format!("Team {} not found", team_id)))?
        };

        // Check team availability
        if !self.is_team_available(&team)? {
            return Err(SklearsError::Other(format!("Team {} is not available", team_id)));
        }

        let deployment = TeamDeployment {
            deployment_id: uuid::Uuid::new_v4().to_string(),
            team_id: team_id.to_string(),
            emergency_id: event.event_id.clone(),
            deployed_at: SystemTime::now(),
            status: DeploymentStatus::Mobilizing,
            assigned_tasks: self.generate_initial_tasks(event, &team)?,
            deployment_location: DeploymentLocation::Remote, // Default
            estimated_deployment_duration: Some(Duration::from_hours(4)),
            actual_deployment_duration: None,
            team_leader: team.members.first().cloned(),
            communication_channel: format!("emergency-{}-{}", event.event_id, team_id),
            performance_metrics: DeploymentPerformanceMetrics::default(),
        };

        // Register deployment
        {
            let mut deployments = self.active_deployments.write()
                .map_err(|_| SklearsError::Other("Failed to acquire deployments lock".into()))?;
            deployments.insert(deployment.deployment_id.clone(), deployment.clone());
        }

        // Update capacity tracker
        {
            let mut tracker = self.capacity_tracker.write()
                .map_err(|_| SklearsError::Other("Failed to acquire capacity tracker lock".into()))?;
            tracker.team_utilization.insert(team_id.to_string(), TeamUtilization {
                team_id: team_id.to_string(),
                current_deployments: 1,
                available_capacity: team.members.len() as f64 - 1.0,
                utilization_percentage: 1.0 / team.members.len() as f64,
                last_updated: SystemTime::now(),
            });
        }

        Ok(deployment)
    }

    /// Assign a task to a deployed team
    pub fn assign_task(&self, deployment_id: &str, task: TaskAssignment) -> SklResult<()> {
        // Verify deployment exists and is active
        {
            let deployments = self.active_deployments.read()
                .map_err(|_| SklearsError::Other("Failed to acquire deployments lock".into()))?;

            if let Some(deployment) = deployments.get(deployment_id) {
                if !matches!(deployment.status, DeploymentStatus::Deployed | DeploymentStatus::Active) {
                    return Err(SklearsError::Other("Deployment is not active".into()));
                }
            } else {
                return Err(SklearsError::InvalidInput("Deployment not found".into()));
            }
        }

        // Register task assignment
        {
            let mut assignments = self.task_assignments.write()
                .map_err(|_| SklearsError::Other("Failed to acquire task assignments lock".into()))?;
            assignments.insert(task.task_id.clone(), task.clone());
        }

        // Add task to deployment
        {
            let mut deployments = self.active_deployments.write()
                .map_err(|_| SklearsError::Other("Failed to acquire deployments lock".into()))?;
            if let Some(deployment) = deployments.get_mut(deployment_id) {
                deployment.assigned_tasks.push(task.task_id.clone());
            }
        }

        Ok(())
    }

    /// Update deployment status
    pub fn update_deployment_status(&self, deployment_id: &str, new_status: DeploymentStatus) -> SklResult<()> {
        let mut deployments = self.active_deployments.write()
            .map_err(|_| SklearsError::Other("Failed to acquire deployments lock".into()))?;

        if let Some(deployment) = deployments.get_mut(deployment_id) {
            let old_status = deployment.status.clone();
            deployment.status = new_status.clone();

            // If standing down, calculate deployment duration
            if matches!(new_status, DeploymentStatus::StoodDown) {
                deployment.actual_deployment_duration = Some(
                    SystemTime::now().duration_since(deployment.deployed_at)
                        .unwrap_or(Duration::from_secs(0))
                );

                // Move to history
                let historical = HistoricalDeployment {
                    deployment: deployment.clone(),
                    completed_at: SystemTime::now(),
                    final_status: new_status,
                    effectiveness_score: 0.85, // Would calculate based on metrics
                    lessons_learned: vec![],
                };

                let mut history = self.deployment_history.write()
                    .map_err(|_| SklearsError::Other("Failed to acquire history lock".into()))?;
                history.push(historical);

                // Update capacity tracker
                let mut tracker = self.capacity_tracker.write()
                    .map_err(|_| SklearsError::Other("Failed to acquire capacity tracker lock".into()))?;
                if let Some(utilization) = tracker.team_utilization.get_mut(&deployment.team_id) {
                    utilization.current_deployments = utilization.current_deployments.saturating_sub(1);
                    utilization.utilization_percentage = utilization.current_deployments as f64 / (utilization.current_deployments as f64 + utilization.available_capacity);
                    utilization.last_updated = SystemTime::now();
                }
            }

            Ok(())
        } else {
            Err(SklearsError::InvalidInput("Deployment not found".into()))
        }
    }

    /// Get active teams
    pub fn get_active_teams(&self) -> SklResult<Vec<String>> {
        let deployments = self.active_deployments.read()
            .map_err(|_| SklearsError::Other("Failed to acquire deployments lock".into()))?;
        Ok(deployments.values().map(|d| d.team_id.clone()).collect())
    }

    /// Get team capacity status
    pub fn get_team_capacity(&self) -> SklResult<TeamCapacityStatus> {
        let tracker = self.capacity_tracker.read()
            .map_err(|_| SklearsError::Other("Failed to acquire capacity tracker lock".into()))?;

        let total_teams = tracker.team_utilization.len() as u64;
        let total_deployments: u64 = tracker.team_utilization.values()
            .map(|u| u.current_deployments as u64)
            .sum();
        let available_teams = tracker.team_utilization.values()
            .filter(|u| u.current_deployments == 0)
            .count() as u64;

        Ok(TeamCapacityStatus {
            total_teams,
            available_teams,
            deployed_teams: total_teams - available_teams,
            total_active_deployments: total_deployments,
            overall_utilization: if total_teams > 0 {
                (total_teams - available_teams) as f64 / total_teams as f64
            } else {
                0.0
            },
            capacity_constraints: self.identify_capacity_constraints(&tracker)?,
        })
    }

    /// Get coordination metrics
    pub fn get_coordination_metrics(&self) -> SklResult<CoordinationMetrics> {
        let metrics = self.coordination_metrics.read()
            .map_err(|_| SklearsError::Other("Failed to acquire metrics lock".into()))?;
        Ok(metrics.clone())
    }

    /// Stand down all teams
    pub fn stand_down_teams(&self) -> SklResult<StandDownResult> {
        let deployment_ids: Vec<String> = {
            let deployments = self.active_deployments.read()
                .map_err(|_| SklearsError::Other("Failed to acquire deployments lock".into()))?;
            deployments.keys().cloned().collect()
        };

        let mut stood_down = 0;
        let mut failed = 0;

        for deployment_id in &deployment_ids {
            match self.update_deployment_status(deployment_id, DeploymentStatus::StandingDown) {
                Ok(_) => {
                    // Complete stand down
                    let _ = self.update_deployment_status(deployment_id, DeploymentStatus::StoodDown);
                    stood_down += 1;
                },
                Err(_) => failed += 1,
            }
        }

        // Clear active deployments
        {
            let mut deployments = self.active_deployments.write()
                .map_err(|_| SklearsError::Other("Failed to acquire deployments lock".into()))?;
            deployments.clear();
        }

        Ok(StandDownResult {
            total_deployments: deployment_ids.len(),
            successful_standdowns: stood_down,
            failed_standdowns: failed,
            standdown_duration: Duration::from_minutes(10), // Estimated
        })
    }

    /// Shutdown coordination system
    pub fn shutdown(&self) -> SklResult<()> {
        // Stand down all teams
        self.stand_down_teams()?;

        // Clear all registries
        {
            let mut assignments = self.task_assignments.write()
                .map_err(|_| SklearsError::Other("Failed to acquire task assignments lock".into()))?;
            assignments.clear();
        }

        Ok(())
    }

    fn setup_response_teams(&self) -> SklResult<()> {
        let mut teams = self.response_teams.write()
            .map_err(|_| SklearsError::Other("Failed to acquire teams lock".into()))?;

        teams.insert("primary_response".to_string(), ResponseTeam {
            team_id: "primary_response".to_string(),
            name: "Primary Response Team".to_string(),
            team_type: TeamType::GeneralResponse,
            specialization: vec!["system_recovery".to_string(), "incident_management".to_string()],
            members: vec![
                TeamMember {
                    member_id: "responder1".to_string(),
                    name: "Alex Johnson".to_string(),
                    role: "Lead Responder".to_string(),
                    skills: vec!["incident_command".to_string(), "system_diagnostics".to_string()],
                    availability: MemberAvailability::OnCall,
                    contact_info: vec!["alex.johnson@company.com".to_string(), "+1-555-0101".to_string()],
                },
                TeamMember {
                    member_id: "responder2".to_string(),
                    name: "Sarah Chen".to_string(),
                    role: "Technical Specialist".to_string(),
                    skills: vec!["database_recovery".to_string(), "network_troubleshooting".to_string()],
                    availability: MemberAvailability::Always,
                    contact_info: vec!["sarah.chen@company.com".to_string(), "+1-555-0102".to_string()],
                },
            ],
            availability: TeamAvailability::Always,
            response_time: Duration::from_minutes(15),
            capabilities: vec!["system_restart".to_string(), "data_recovery".to_string(), "incident_coordination".to_string()],
            current_capacity: 2.0,
            max_capacity: 2.0,
            performance_metrics: TeamPerformanceMetrics::default(),
        });

        teams.insert("security_response".to_string(), ResponseTeam {
            team_id: "security_response".to_string(),
            name: "Security Response Team".to_string(),
            team_type: TeamType::SecuritySpecialist,
            specialization: vec!["security_incidents".to_string(), "breach_response".to_string(), "forensics".to_string()],
            members: vec![
                TeamMember {
                    member_id: "security1".to_string(),
                    name: "Mike Davis".to_string(),
                    role: "Security Lead".to_string(),
                    skills: vec!["threat_analysis".to_string(), "incident_response".to_string()],
                    availability: MemberAvailability::OnCall,
                    contact_info: vec!["mike.davis@company.com".to_string(), "+1-555-0201".to_string()],
                },
                TeamMember {
                    member_id: "security2".to_string(),
                    name: "Emma Wilson".to_string(),
                    role: "Forensics Specialist".to_string(),
                    skills: vec!["digital_forensics".to_string(), "malware_analysis".to_string()],
                    availability: MemberAvailability::BusinessHours,
                    contact_info: vec!["emma.wilson@company.com".to_string(), "+1-555-0202".to_string()],
                },
            ],
            availability: TeamAvailability::OnCall,
            response_time: Duration::from_minutes(30),
            capabilities: vec!["forensics".to_string(), "threat_mitigation".to_string(), "security_assessment".to_string()],
            current_capacity: 2.0,
            max_capacity: 2.0,
            performance_metrics: TeamPerformanceMetrics::default(),
        });

        teams.insert("infrastructure_team".to_string(), ResponseTeam {
            team_id: "infrastructure_team".to_string(),
            name: "Infrastructure Response Team".to_string(),
            team_type: TeamType::InfrastructureSpecialist,
            specialization: vec!["infrastructure_recovery".to_string(), "capacity_scaling".to_string()],
            members: vec![
                TeamMember {
                    member_id: "infra1".to_string(),
                    name: "David Kim".to_string(),
                    role: "Infrastructure Lead".to_string(),
                    skills: vec!["cloud_infrastructure".to_string(), "automation".to_string()],
                    availability: MemberAvailability::Always,
                    contact_info: vec!["david.kim@company.com".to_string(), "+1-555-0301".to_string()],
                },
            ],
            availability: TeamAvailability::Always,
            response_time: Duration::from_minutes(20),
            capabilities: vec!["resource_scaling".to_string(), "infrastructure_recovery".to_string()],
            current_capacity: 1.0,
            max_capacity: 1.0,
            performance_metrics: TeamPerformanceMetrics::default(),
        });

        Ok(())
    }

    fn initialize_capacity_tracking(&self) -> SklResult<()> {
        let mut tracker = self.capacity_tracker.write()
            .map_err(|_| SklearsError::Other("Failed to acquire capacity tracker lock".into()))?;

        let teams = self.response_teams.read()
            .map_err(|_| SklearsError::Other("Failed to acquire teams lock".into()))?;

        for (team_id, team) in teams.iter() {
            tracker.team_utilization.insert(team_id.clone(), TeamUtilization {
                team_id: team_id.clone(),
                current_deployments: 0,
                available_capacity: team.members.len() as f64,
                utilization_percentage: 0.0,
                last_updated: SystemTime::now(),
            });
        }

        Ok(())
    }

    fn select_teams_for_emergency(&self, event: &EmergencyEvent) -> SklResult<Vec<String>> {
        let teams = self.response_teams.read()
            .map_err(|_| SklearsError::Other("Failed to acquire teams lock".into()))?;

        let mut selected_teams = Vec::new();

        // Always include primary response team
        selected_teams.push("primary_response".to_string());

        // Add specialized teams based on emergency type
        match event.emergency_type {
            EmergencyType::SecurityIncident | EmergencyType::DataBreach => {
                selected_teams.push("security_response".to_string());
            },
            EmergencyType::SystemFailure | EmergencyType::ResourceExhaustion => {
                selected_teams.push("infrastructure_team".to_string());
            },
            _ => {}
        }

        // Filter available teams
        let available_teams: Vec<String> = selected_teams.into_iter()
            .filter(|team_id| {
                teams.get(team_id)
                    .map(|team| self.is_team_available(team).unwrap_or(false))
                    .unwrap_or(false)
            })
            .collect();

        Ok(available_teams)
    }

    fn is_team_available(&self, team: &ResponseTeam) -> SklResult<bool> {
        let tracker = self.capacity_tracker.read()
            .map_err(|_| SklearsError::Other("Failed to acquire capacity tracker lock".into()))?;

        if let Some(utilization) = tracker.team_utilization.get(&team.team_id) {
            Ok(utilization.current_deployments == 0 && utilization.available_capacity > 0.0)
        } else {
            Ok(true) // If not tracked, assume available
        }
    }

    fn generate_initial_tasks(&self, event: &EmergencyEvent, team: &ResponseTeam) -> SklResult<Vec<String>> {
        let mut tasks = Vec::new();

        // Generate tasks based on team specialization and emergency type
        match event.emergency_type {
            EmergencyType::SystemFailure => {
                if team.capabilities.contains(&"system_restart".to_string()) {
                    tasks.push("system_diagnostics".to_string());
                    tasks.push("system_recovery".to_string());
                }
            },
            EmergencyType::SecurityIncident => {
                if team.capabilities.contains(&"forensics".to_string()) {
                    tasks.push("threat_assessment".to_string());
                    tasks.push("containment".to_string());
                }
            },
            _ => {
                tasks.push("initial_assessment".to_string());
            }
        }

        Ok(tasks)
    }

    fn calculate_response_capacity(&self, deployments: &[TeamDeployment]) -> SklResult<ResponseCapacity> {
        let teams = self.response_teams.read()
            .map_err(|_| SklearsError::Other("Failed to acquire teams lock".into()))?;

        let total_members: usize = deployments.iter()
            .filter_map(|d| teams.get(&d.team_id))
            .map(|t| t.members.len())
            .sum();

        let combined_capabilities: Vec<String> = deployments.iter()
            .filter_map(|d| teams.get(&d.team_id))
            .flat_map(|t| t.capabilities.iter().cloned())
            .collect();

        Ok(ResponseCapacity {
            total_deployed_members: total_members,
            available_capabilities: combined_capabilities,
            estimated_response_effectiveness: 0.85, // Would calculate based on team performance
            coordination_complexity: deployments.len() as f64 * 0.1, // Simple heuristic
        })
    }

    fn identify_capacity_constraints(&self, tracker: &TeamCapacityTracker) -> SklResult<Vec<CapacityConstraint>> {
        let mut constraints = Vec::new();

        for utilization in tracker.team_utilization.values() {
            if utilization.utilization_percentage > 0.8 {
                constraints.push(CapacityConstraint {
                    constraint_type: ConstraintType::HighUtilization,
                    affected_team: utilization.team_id.clone(),
                    severity: if utilization.utilization_percentage > 0.95 {
                        ConstraintSeverity::Critical
                    } else {
                        ConstraintSeverity::High
                    },
                    description: format!("Team {} has high utilization: {:.1}%",
                        utilization.team_id, utilization.utilization_percentage * 100.0),
                });
            }
        }

        Ok(constraints)
    }
}

/// Response team structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTeam {
    pub team_id: String,
    pub name: String,
    pub team_type: TeamType,
    pub specialization: Vec<String>,
    pub members: Vec<TeamMember>,
    pub availability: TeamAvailability,
    pub response_time: Duration,
    pub capabilities: Vec<String>,
    pub current_capacity: f64,
    pub max_capacity: f64,
    pub performance_metrics: TeamPerformanceMetrics,
}

/// Team types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TeamType {
    GeneralResponse,
    SecuritySpecialist,
    InfrastructureSpecialist,
    DatabaseSpecialist,
    NetworkSpecialist,
    ApplicationSpecialist,
}

/// Team member information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamMember {
    pub member_id: String,
    pub name: String,
    pub role: String,
    pub skills: Vec<String>,
    pub availability: MemberAvailability,
    pub contact_info: Vec<String>,
}

/// Member availability status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemberAvailability {
    Always,
    BusinessHours,
    OnCall,
    Emergency,
    Unavailable,
}

/// Team availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TeamAvailability {
    Always,
    BusinessHours,
    OnCall,
    Emergency,
}

/// Team deployment tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamDeployment {
    pub deployment_id: String,
    pub team_id: String,
    pub emergency_id: String,
    pub deployed_at: SystemTime,
    pub status: DeploymentStatus,
    pub assigned_tasks: Vec<String>,
    pub deployment_location: DeploymentLocation,
    pub estimated_deployment_duration: Option<Duration>,
    pub actual_deployment_duration: Option<Duration>,
    pub team_leader: Option<String>,
    pub communication_channel: String,
    pub performance_metrics: DeploymentPerformanceMetrics,
}

/// Deployment status tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeploymentStatus {
    Mobilizing,
    Deployed,
    Active,
    StandingDown,
    StoodDown,
    Failed,
}

/// Deployment location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentLocation {
    OnSite,
    Remote,
    Hybrid,
    DataCenter,
    CloudRegion(String),
}

/// Task assignment tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAssignment {
    pub task_id: String,
    pub deployment_id: String,
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub description: String,
    pub assigned_to: Vec<String>,
    pub estimated_duration: Duration,
    pub actual_duration: Option<Duration>,
    pub status: TaskStatus,
    pub dependencies: Vec<String>,
    pub progress: f64,
}

/// Task types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Assessment,
    Diagnostics,
    Recovery,
    Mitigation,
    Monitoring,
    Communication,
    Coordination,
}

/// Task priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Assigned,
    InProgress,
    Blocked,
    Completed,
    Failed,
    Cancelled,
}

/// Team capacity tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamCapacityTracker {
    pub team_utilization: HashMap<String, TeamUtilization>,
    pub last_updated: SystemTime,
}

impl TeamCapacityTracker {
    pub fn new() -> Self {
        Self {
            team_utilization: HashMap::new(),
            last_updated: SystemTime::now(),
        }
    }
}

/// Team utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamUtilization {
    pub team_id: String,
    pub current_deployments: u32,
    pub available_capacity: f64,
    pub utilization_percentage: f64,
    pub last_updated: SystemTime,
}

/// Mobilization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobilizationResult {
    pub mobilization_id: String,
    pub emergency_id: String,
    pub teams_mobilized: usize,
    pub deployment_ids: Vec<String>,
    pub mobilization_time: Duration,
    pub success: bool,
    pub estimated_response_capacity: ResponseCapacity,
}

/// Response capacity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCapacity {
    pub total_deployed_members: usize,
    pub available_capabilities: Vec<String>,
    pub estimated_response_effectiveness: f64,
    pub coordination_complexity: f64,
}

/// Team capacity status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamCapacityStatus {
    pub total_teams: u64,
    pub available_teams: u64,
    pub deployed_teams: u64,
    pub total_active_deployments: u64,
    pub overall_utilization: f64,
    pub capacity_constraints: Vec<CapacityConstraint>,
}

/// Capacity constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityConstraint {
    pub constraint_type: ConstraintType,
    pub affected_team: String,
    pub severity: ConstraintSeverity,
    pub description: String,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    HighUtilization,
    InsufficientSkills,
    UnavailableMembers,
    ResourceLimits,
}

/// Constraint severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Stand down operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandDownResult {
    pub total_deployments: usize,
    pub successful_standdowns: usize,
    pub failed_standdowns: usize,
    pub standdown_duration: Duration,
}

/// Historical deployment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalDeployment {
    pub deployment: TeamDeployment,
    pub completed_at: SystemTime,
    pub final_status: DeploymentStatus,
    pub effectiveness_score: f64,
    pub lessons_learned: Vec<String>,
}

/// Team performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamPerformanceMetrics {
    pub response_time_average: Duration,
    pub task_completion_rate: f64,
    pub effectiveness_score: f64,
    pub deployment_count: u32,
    pub total_response_time: Duration,
}

impl Default for TeamPerformanceMetrics {
    fn default() -> Self {
        Self {
            response_time_average: Duration::from_minutes(20),
            task_completion_rate: 0.85,
            effectiveness_score: 0.78,
            deployment_count: 0,
            total_response_time: Duration::from_secs(0),
        }
    }
}

/// Deployment performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentPerformanceMetrics {
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub average_task_completion_time: Duration,
    pub team_coordination_score: f64,
    pub communication_effectiveness: f64,
}

impl Default for DeploymentPerformanceMetrics {
    fn default() -> Self {
        Self {
            tasks_completed: 0,
            tasks_failed: 0,
            average_task_completion_time: Duration::from_hours(1),
            team_coordination_score: 0.80,
            communication_effectiveness: 0.75,
        }
    }
}

/// Coordination metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMetrics {
    pub total_mobilizations: u64,
    pub teams_mobilized: u64,
    pub average_mobilization_time: Duration,
    pub deployment_success_rate: f64,
    pub coordination_effectiveness: f64,
    pub last_mobilization: Option<SystemTime>,
}

impl CoordinationMetrics {
    pub fn new() -> Self {
        Self {
            total_mobilizations: 0,
            teams_mobilized: 0,
            average_mobilization_time: Duration::from_minutes(15),
            deployment_success_rate: 0.90,
            coordination_effectiveness: 0.85,
            last_mobilization: None,
        }
    }
}

impl Default for ResponseTeamCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_coordinator_creation() {
        let coordinator = ResponseTeamCoordinator::new();
        assert!(coordinator.initialize().is_ok());
    }

    #[test]
    fn test_team_mobilization() {
        let coordinator = ResponseTeamCoordinator::new();
        coordinator.initialize().unwrap();

        let event = EmergencyEvent {
            event_id: "emergency-001".to_string(),
            emergency_type: EmergencyType::SystemFailure,
            severity: EmergencySeverity::Critical,
            title: "System Down".to_string(),
            description: "Primary system is unresponsive".to_string(),
            source: "health_monitor".to_string(),
            timestamp: SystemTime::now(),
            affected_systems: vec!["primary_system".to_string()],
            estimated_impact: super::super::detection::EmergencyImpact {
                user_impact: super::super::detection::UserImpact::High,
                business_impact: super::super::detection::BusinessImpact::High,
                system_impact: super::super::detection::SystemImpact::Critical,
                financial_impact: Some(10000.0),
            },
            estimated_impact_duration: Some(Duration::from_hours(2)),
            detected_by: "detector".to_string(),
            context: std::collections::HashMap::new(),
            related_events: vec![],
            urgency: super::super::detection::Urgency::High,
            requires_immediate_action: true,
        };

        let result = coordinator.mobilize_teams(&event);
        assert!(result.is_ok());

        let mobilization = result.unwrap();
        assert!(mobilization.teams_mobilized > 0);
        assert!(!mobilization.deployment_ids.is_empty());
    }

    #[test]
    fn test_team_capacity_tracking() {
        let coordinator = ResponseTeamCoordinator::new();
        coordinator.initialize().unwrap();

        let capacity = coordinator.get_team_capacity().unwrap();
        assert!(capacity.total_teams > 0);
        assert_eq!(capacity.available_teams, capacity.total_teams);
        assert_eq!(capacity.deployed_teams, 0);
    }

    #[test]
    fn test_deployment_status_update() {
        let coordinator = ResponseTeamCoordinator::new();
        coordinator.initialize().unwrap();

        let event = EmergencyEvent {
            event_id: "emergency-002".to_string(),
            emergency_type: EmergencyType::SecurityIncident,
            severity: EmergencySeverity::High,
            title: "Security Breach".to_string(),
            description: "Unauthorized access detected".to_string(),
            source: "security_monitor".to_string(),
            timestamp: SystemTime::now(),
            affected_systems: vec!["user_database".to_string()],
            estimated_impact: super::super::detection::EmergencyImpact {
                user_impact: super::super::detection::UserImpact::Critical,
                business_impact: super::super::detection::BusinessImpact::Critical,
                system_impact: super::super::detection::SystemImpact::High,
                financial_impact: Some(50000.0),
            },
            estimated_impact_duration: Some(Duration::from_hours(4)),
            detected_by: "security_detector".to_string(),
            context: std::collections::HashMap::new(),
            related_events: vec![],
            urgency: super::super::detection::Urgency::Critical,
            requires_immediate_action: true,
        };

        let deployment = coordinator.deploy_team("security_response", &event).unwrap();
        let result = coordinator.update_deployment_status(&deployment.deployment_id, DeploymentStatus::Active);
        assert!(result.is_ok());
    }
}