//! Role-Based Access Control (RBAC) for model explanations
//!
//! This module implements comprehensive role-based access control for explanation systems,
//! allowing fine-grained control over who can view, generate, modify, and manage explanations.

use crate::{SklResult, SklearsError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Access levels for explanation resources
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessLevel {
    /// No access to the resource
    None,
    /// Read-only access (view explanations)
    Read,
    /// Write access (generate explanations)
    Write,
    /// Administrative access (manage explanations, users, roles)
    Admin,
    /// System-level access (full control)
    System,
}

impl AccessLevel {
    /// Check if this access level implies another level
    pub fn implies(&self, other: &AccessLevel) -> bool {
        match self {
            AccessLevel::System => true,
            AccessLevel::Admin => matches!(
                other,
                AccessLevel::Read | AccessLevel::Write | AccessLevel::Admin
            ),
            AccessLevel::Write => matches!(other, AccessLevel::Read | AccessLevel::Write),
            AccessLevel::Read => matches!(other, AccessLevel::Read),
            AccessLevel::None => matches!(other, AccessLevel::None),
        }
    }
}

/// Specific permissions within the explanation system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    // Explanation operations
    /// ViewExplanations

    /// ViewExplanations
    ViewExplanations,
    /// GenerateExplanations

    /// GenerateExplanations
    GenerateExplanations,
    /// ModifyExplanations

    /// ModifyExplanations
    ModifyExplanations,
    /// DeleteExplanations

    /// DeleteExplanations
    DeleteExplanations,
    /// ExportExplanations

    /// ExportExplanations
    ExportExplanations,

    // Model operations
    /// ViewModels

    /// ViewModels
    ViewModels,
    /// RegisterModels

    /// RegisterModels
    RegisterModels,
    /// ModifyModels
    ModifyModels,
    /// DeleteModels
    DeleteModels,

    // Data operations
    /// ViewData
    ViewData,
    /// UploadData
    UploadData,
    /// ModifyData
    ModifyData,
    /// DeleteData
    DeleteData,

    // User management
    /// ViewUsers
    ViewUsers,
    /// CreateUsers
    CreateUsers,
    /// ModifyUsers
    ModifyUsers,
    /// DeleteUsers
    DeleteUsers,

    // Role management
    /// ViewRoles
    ViewRoles,
    /// CreateRoles
    CreateRoles,
    /// ModifyRoles
    ModifyRoles,
    /// DeleteRoles
    DeleteRoles,

    // System administration
    /// ViewAuditLogs
    ViewAuditLogs,
    /// ModifySystemSettings
    ModifySystemSettings,
    /// ManageCompliance
    ManageCompliance,
    /// MonitorQuality
    MonitorQuality,
}

/// Set of permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionSet {
    permissions: HashSet<Permission>,
}

impl PermissionSet {
    /// Create a new empty permission set
    pub fn new() -> Self {
        Self {
            permissions: HashSet::new(),
        }
    }

    /// Create a permission set from a vector of permissions
    pub fn from_permissions(permissions: Vec<Permission>) -> Self {
        Self {
            permissions: permissions.into_iter().collect(),
        }
    }

    /// Add a permission to the set
    pub fn add_permission(&mut self, permission: Permission) {
        self.permissions.insert(permission);
    }

    /// Remove a permission from the set
    pub fn remove_permission(&mut self, permission: &Permission) {
        self.permissions.remove(permission);
    }

    /// Check if the set contains a specific permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
    }

    /// Get all permissions in the set
    pub fn get_permissions(&self) -> &HashSet<Permission> {
        &self.permissions
    }

    /// Create a permission set for a specific access level
    pub fn for_access_level(level: AccessLevel) -> Self {
        let permissions = match level {
            AccessLevel::None => vec![],
            AccessLevel::Read => vec![
                Permission::ViewExplanations,
                Permission::ViewModels,
                Permission::ViewData,
            ],
            AccessLevel::Write => vec![
                Permission::ViewExplanations,
                Permission::GenerateExplanations,
                Permission::ModifyExplanations,
                Permission::ExportExplanations,
                Permission::ViewModels,
                Permission::ViewData,
                Permission::UploadData,
            ],
            AccessLevel::Admin => vec![
                Permission::ViewExplanations,
                Permission::GenerateExplanations,
                Permission::ModifyExplanations,
                Permission::DeleteExplanations,
                Permission::ExportExplanations,
                Permission::ViewModels,
                Permission::RegisterModels,
                Permission::ModifyModels,
                Permission::ViewData,
                Permission::UploadData,
                Permission::ModifyData,
                Permission::ViewUsers,
                Permission::CreateUsers,
                Permission::ModifyUsers,
                Permission::ViewRoles,
                Permission::ViewAuditLogs,
                Permission::MonitorQuality,
            ],
            AccessLevel::System => Permission::all_permissions(),
        };

        Self::from_permissions(permissions)
    }
}

impl Permission {
    /// Get all available permissions
    pub fn all_permissions() -> Vec<Permission> {
        vec![
            Permission::ViewExplanations,
            Permission::GenerateExplanations,
            Permission::ModifyExplanations,
            Permission::DeleteExplanations,
            Permission::ExportExplanations,
            Permission::ViewModels,
            Permission::RegisterModels,
            Permission::ModifyModels,
            Permission::DeleteModels,
            Permission::ViewData,
            Permission::UploadData,
            Permission::ModifyData,
            Permission::DeleteData,
            Permission::ViewUsers,
            Permission::CreateUsers,
            Permission::ModifyUsers,
            Permission::DeleteUsers,
            Permission::ViewRoles,
            Permission::CreateRoles,
            Permission::ModifyRoles,
            Permission::DeleteRoles,
            Permission::ViewAuditLogs,
            Permission::ModifySystemSettings,
            Permission::ManageCompliance,
            Permission::MonitorQuality,
        ]
    }
}

impl Default for PermissionSet {
    fn default() -> Self {
        Self::new()
    }
}

/// User role within the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Role description
    pub description: String,
    /// Permissions granted by this role
    pub permissions: PermissionSet,
    /// When the role was created
    pub created_at: DateTime<Utc>,
    /// When the role was last modified
    pub modified_at: DateTime<Utc>,
}

impl Role {
    /// Create a new role
    pub fn new(name: String, description: String) -> Self {
        let now = Utc::now();
        Self {
            name,
            description,
            permissions: PermissionSet::new(),
            created_at: now,
            modified_at: now,
        }
    }

    /// Create a predefined role
    pub fn predefined(name: &str) -> SklResult<Self> {
        let (name, description, access_level) = match name {
            "viewer" => (
                "Viewer",
                "Can view explanations and models",
                AccessLevel::Read,
            ),
            "analyst" => (
                "Analyst",
                "Can generate and modify explanations",
                AccessLevel::Write,
            ),
            "admin" => (
                "Administrator",
                "Can manage users, roles, and system settings",
                AccessLevel::Admin,
            ),
            "system" => (
                "System Administrator",
                "Full system access",
                AccessLevel::System,
            ),
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "role_name".to_string(),
                    reason: format!("Unknown predefined role: {}", name),
                })
            }
        };

        let mut role = Self::new(name.to_string(), description.to_string());
        role.permissions = PermissionSet::for_access_level(access_level);
        Ok(role)
    }

    /// Add a permission to the role
    pub fn add_permission(&mut self, permission: Permission) {
        self.permissions.add_permission(permission);
        self.modified_at = Utc::now();
    }

    /// Remove a permission from the role
    pub fn remove_permission(&mut self, permission: &Permission) {
        self.permissions.remove_permission(permission);
        self.modified_at = Utc::now();
    }

    /// Check if the role has a specific permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.has_permission(permission)
    }
}

/// User group for organizing users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserGroup {
    /// Group name
    pub name: String,
    /// Group description
    pub description: String,
    /// Roles assigned to this group
    pub roles: Vec<String>,
    /// When the group was created
    pub created_at: DateTime<Utc>,
}

impl UserGroup {
    /// Create a new user group
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            roles: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Add a role to the group
    pub fn add_role(&mut self, role_name: String) {
        if !self.roles.contains(&role_name) {
            self.roles.push(role_name);
        }
    }

    /// Remove a role from the group
    pub fn remove_role(&mut self, role_name: &str) {
        self.roles.retain(|r| r != role_name);
    }
}

/// System user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// User ID
    pub id: String,
    /// Username
    pub username: String,
    /// User email
    pub email: String,
    /// User's full name
    pub full_name: String,
    /// Directly assigned roles
    pub roles: Vec<String>,
    /// Groups the user belongs to
    pub groups: Vec<String>,
    /// User-specific permissions (overrides)
    pub permissions: PermissionSet,
    /// Whether the user is active
    pub active: bool,
    /// When the user was created
    pub created_at: DateTime<Utc>,
    /// When the user last logged in
    pub last_login: Option<DateTime<Utc>>,
}

impl User {
    /// Create a new user
    pub fn new(id: String, username: String, email: String, full_name: String) -> Self {
        Self {
            id,
            username,
            email,
            full_name,
            roles: Vec::new(),
            groups: Vec::new(),
            permissions: PermissionSet::new(),
            active: true,
            created_at: Utc::now(),
            last_login: None,
        }
    }

    /// Add a role to the user
    pub fn add_role(&mut self, role_name: String) {
        if !self.roles.contains(&role_name) {
            self.roles.push(role_name);
        }
    }

    /// Remove a role from the user
    pub fn remove_role(&mut self, role_name: &str) {
        self.roles.retain(|r| r != role_name);
    }

    /// Add the user to a group
    pub fn add_to_group(&mut self, group_name: String) {
        if !self.groups.contains(&group_name) {
            self.groups.push(group_name);
        }
    }

    /// Remove the user from a group
    pub fn remove_from_group(&mut self, group_name: &str) {
        self.groups.retain(|g| g != group_name);
    }

    /// Update last login time
    pub fn update_last_login(&mut self) {
        self.last_login = Some(Utc::now());
    }
}

/// Security context for the current operation
#[derive(Debug, Clone)]
pub struct SecurityContext {
    /// Current user
    pub user: User,
    /// Effective permissions (computed from roles, groups, and overrides)
    pub effective_permissions: PermissionSet,
    /// Session information
    pub session_id: String,
    /// Request timestamp
    pub timestamp: DateTime<Utc>,
}

impl SecurityContext {
    /// Create a security context for a user
    pub fn for_user(user: User, role_manager: &RoleManager, session_id: String) -> Self {
        let effective_permissions = role_manager.compute_effective_permissions(&user);

        Self {
            user,
            effective_permissions,
            session_id,
            timestamp: Utc::now(),
        }
    }

    /// Check if the current context has a specific permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.effective_permissions.has_permission(permission)
    }

    /// Require a specific permission, returning an error if not granted
    pub fn require_permission(&self, permission: &Permission) -> SklResult<()> {
        if self.has_permission(permission) {
            Ok(())
        } else {
            Err(SklearsError::InvalidOperation(format!(
                "User {} does not have permission: {:?}",
                self.user.username, permission
            )))
        }
    }
}

/// Role manager for handling RBAC operations
#[derive(Debug)]
pub struct RoleManager {
    /// All system roles
    roles: HashMap<String, Role>,
    /// All user groups
    groups: HashMap<String, UserGroup>,
    /// All system users
    users: HashMap<String, User>,
}

impl RoleManager {
    /// Create a new role manager
    pub fn new() -> Self {
        let mut manager = Self {
            roles: HashMap::new(),
            groups: HashMap::new(),
            users: HashMap::new(),
        };

        // Create default roles
        if let Ok(viewer) = Role::predefined("viewer") {
            manager.roles.insert("viewer".to_string(), viewer);
        }
        if let Ok(analyst) = Role::predefined("analyst") {
            manager.roles.insert("analyst".to_string(), analyst);
        }
        if let Ok(admin) = Role::predefined("admin") {
            manager.roles.insert("admin".to_string(), admin);
        }
        if let Ok(system) = Role::predefined("system") {
            manager.roles.insert("system".to_string(), system);
        }

        manager
    }

    /// Add a role to the system
    pub fn add_role(&mut self, role: Role) -> SklResult<()> {
        if self.roles.contains_key(&role.name) {
            return Err(SklearsError::InvalidParameter {
                name: "role_name".to_string(),
                reason: format!("Role '{}' already exists", role.name),
            });
        }
        self.roles.insert(role.name.clone(), role);
        Ok(())
    }

    /// Get a role by name
    pub fn get_role(&self, name: &str) -> Option<&Role> {
        self.roles.get(name)
    }

    /// Remove a role
    pub fn remove_role(&mut self, name: &str) -> SklResult<()> {
        if !self.roles.contains_key(name) {
            return Err(SklearsError::InvalidParameter {
                name: "role_name".to_string(),
                reason: format!("Role '{}' does not exist", name),
            });
        }
        self.roles.remove(name);
        Ok(())
    }

    /// Add a user group
    pub fn add_group(&mut self, group: UserGroup) -> SklResult<()> {
        if self.groups.contains_key(&group.name) {
            return Err(SklearsError::InvalidParameter {
                name: "group_name".to_string(),
                reason: format!("Group '{}' already exists", group.name),
            });
        }
        self.groups.insert(group.name.clone(), group);
        Ok(())
    }

    /// Get a group by name
    pub fn get_group(&self, name: &str) -> Option<&UserGroup> {
        self.groups.get(name)
    }

    /// Add a user to the system
    pub fn add_user(&mut self, user: User) -> SklResult<()> {
        if self.users.contains_key(&user.id) {
            return Err(SklearsError::InvalidParameter {
                name: "user_id".to_string(),
                reason: format!("User '{}' already exists", user.id),
            });
        }
        self.users.insert(user.id.clone(), user);
        Ok(())
    }

    /// Get a user by ID
    pub fn get_user(&self, id: &str) -> Option<&User> {
        self.users.get(id)
    }

    /// Get a user by username
    pub fn get_user_by_username(&self, username: &str) -> Option<&User> {
        self.users.values().find(|user| user.username == username)
    }

    /// Compute effective permissions for a user
    pub fn compute_effective_permissions(&self, user: &User) -> PermissionSet {
        let mut effective_permissions = PermissionSet::new();

        // Add permissions from directly assigned roles
        for role_name in &user.roles {
            if let Some(role) = self.roles.get(role_name) {
                for permission in role.permissions.get_permissions() {
                    effective_permissions.add_permission(permission.clone());
                }
            }
        }

        // Add permissions from group roles
        for group_name in &user.groups {
            if let Some(group) = self.groups.get(group_name) {
                for role_name in &group.roles {
                    if let Some(role) = self.roles.get(role_name) {
                        for permission in role.permissions.get_permissions() {
                            effective_permissions.add_permission(permission.clone());
                        }
                    }
                }
            }
        }

        // Add user-specific permission overrides
        for permission in user.permissions.get_permissions() {
            effective_permissions.add_permission(permission.clone());
        }

        effective_permissions
    }

    /// Check if a user has a specific permission
    pub fn user_has_permission(&self, user: &User, permission: &Permission) -> bool {
        let effective_permissions = self.compute_effective_permissions(user);
        effective_permissions.has_permission(permission)
    }

    /// Get all roles
    pub fn get_all_roles(&self) -> Vec<&Role> {
        self.roles.values().collect()
    }

    /// Get all groups
    pub fn get_all_groups(&self) -> Vec<&UserGroup> {
        self.groups.values().collect()
    }

    /// Get all users
    pub fn get_all_users(&self) -> Vec<&User> {
        self.users.values().collect()
    }
}

impl Default for RoleManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Whether RBAC is enabled
    pub enabled: bool,
    /// Default role for new users
    pub default_role: String,
    /// Session timeout in seconds
    pub session_timeout: u64,
    /// Whether to log access control events
    pub log_access_events: bool,
    /// Whether to enforce strong authentication
    pub enforce_mfa: bool,
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_role: "viewer".to_string(),
            session_timeout: 3600, // 1 hour
            log_access_events: true,
            enforce_mfa: false,
        }
    }
}

/// Access control system
#[derive(Debug)]
pub struct AccessControl {
    /// Configuration
    config: AccessControlConfig,
    /// Role manager
    role_manager: RoleManager,
    /// Active sessions
    active_sessions: HashMap<String, SecurityContext>,
}

impl AccessControl {
    /// Create a new access control system
    pub fn new(config: AccessControlConfig) -> Self {
        Self {
            config,
            role_manager: RoleManager::new(),
            active_sessions: HashMap::new(),
        }
    }

    /// Create a session for a user
    pub fn create_session(
        &mut self,
        user_id: &str,
        session_id: String,
    ) -> SklResult<SecurityContext> {
        if !self.config.enabled {
            // Create a permissive context when RBAC is disabled
            let mut admin_user = User::new(
                user_id.to_string(),
                "system".to_string(),
                "system@local".to_string(),
                "System User".to_string(),
            );
            admin_user.add_role("system".to_string());

            let context = SecurityContext::for_user(admin_user, &self.role_manager, session_id);
            return Ok(context);
        }

        let user =
            self.role_manager
                .get_user(user_id)
                .ok_or_else(|| SklearsError::InvalidParameter {
                    name: "user_id".to_string(),
                    reason: format!("User '{}' not found", user_id),
                })?;

        if !user.active {
            return Err(SklearsError::InvalidOperation(format!(
                "User '{}' is inactive",
                user_id
            )));
        }

        let mut user = user.clone();
        user.update_last_login();

        let context = SecurityContext::for_user(user, &self.role_manager, session_id.clone());
        self.active_sessions.insert(session_id, context.clone());

        Ok(context)
    }

    /// Get a session by ID
    pub fn get_session(&self, session_id: &str) -> Option<&SecurityContext> {
        self.active_sessions.get(session_id)
    }

    /// Remove a session
    pub fn remove_session(&mut self, session_id: &str) {
        self.active_sessions.remove(session_id);
    }

    /// Get the role manager
    pub fn role_manager(&self) -> &RoleManager {
        &self.role_manager
    }

    /// Get the role manager mutably
    pub fn role_manager_mut(&mut self) -> &mut RoleManager {
        &mut self.role_manager
    }

    /// Check if RBAC is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

/// Secure wrapper for explanation execution with access control
#[derive(Debug)]
pub struct SecureExplanationExecutor {
    /// Access control system
    access_control: AccessControl,
}

impl SecureExplanationExecutor {
    /// Create a new secure executor
    pub fn new(access_control: AccessControl) -> Self {
        Self { access_control }
    }

    /// Execute an explanation operation with security checks
    pub fn execute_with_security<T, F>(
        &self,
        session_id: &str,
        required_permission: Permission,
        operation: F,
    ) -> SklResult<T>
    where
        F: FnOnce() -> SklResult<T>,
    {
        if !self.access_control.is_enabled() {
            return operation();
        }

        let context = self
            .access_control
            .get_session(session_id)
            .ok_or_else(|| SklearsError::InvalidOperation("Invalid session".to_string()))?;

        context.require_permission(&required_permission)?;

        operation()
    }

    /// Get access control system
    pub fn access_control(&self) -> &AccessControl {
        &self.access_control
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_level_implies() {
        assert!(AccessLevel::System.implies(&AccessLevel::Admin));
        assert!(AccessLevel::System.implies(&AccessLevel::Write));
        assert!(AccessLevel::System.implies(&AccessLevel::Read));
        assert!(AccessLevel::Admin.implies(&AccessLevel::Write));
        assert!(AccessLevel::Admin.implies(&AccessLevel::Read));
        assert!(AccessLevel::Write.implies(&AccessLevel::Read));
        assert!(!AccessLevel::Read.implies(&AccessLevel::Write));
        assert!(!AccessLevel::None.implies(&AccessLevel::Read));
    }

    #[test]
    fn test_permission_set() {
        let mut perm_set = PermissionSet::new();
        assert!(!perm_set.has_permission(&Permission::ViewExplanations));

        perm_set.add_permission(Permission::ViewExplanations);
        assert!(perm_set.has_permission(&Permission::ViewExplanations));

        perm_set.remove_permission(&Permission::ViewExplanations);
        assert!(!perm_set.has_permission(&Permission::ViewExplanations));
    }

    #[test]
    fn test_role_creation() {
        let role = Role::new("test_role".to_string(), "Test role".to_string());
        assert_eq!(role.name, "test_role");
        assert_eq!(role.description, "Test role");
    }

    #[test]
    fn test_predefined_roles() {
        let viewer = Role::predefined("viewer").unwrap();
        assert!(viewer.has_permission(&Permission::ViewExplanations));
        assert!(!viewer.has_permission(&Permission::GenerateExplanations));

        let analyst = Role::predefined("analyst").unwrap();
        assert!(analyst.has_permission(&Permission::ViewExplanations));
        assert!(analyst.has_permission(&Permission::GenerateExplanations));

        let admin = Role::predefined("admin").unwrap();
        assert!(admin.has_permission(&Permission::ViewExplanations));
        assert!(admin.has_permission(&Permission::ViewUsers));
    }

    #[test]
    fn test_user_creation() {
        let mut user = User::new(
            "user1".to_string(),
            "testuser".to_string(),
            "test@example.com".to_string(),
            "Test User".to_string(),
        );

        assert_eq!(user.id, "user1");
        assert_eq!(user.username, "testuser");
        assert!(user.active);

        user.add_role("viewer".to_string());
        assert!(user.roles.contains(&"viewer".to_string()));
    }

    #[test]
    fn test_role_manager() {
        let mut manager = RoleManager::new();

        // Test predefined roles exist
        assert!(manager.get_role("viewer").is_some());
        assert!(manager.get_role("analyst").is_some());
        assert!(manager.get_role("admin").is_some());

        // Test adding custom role
        let custom_role = Role::new("custom".to_string(), "Custom role".to_string());
        manager.add_role(custom_role).unwrap();
        assert!(manager.get_role("custom").is_some());

        // Test adding user
        let user = User::new(
            "user1".to_string(),
            "testuser".to_string(),
            "test@example.com".to_string(),
            "Test User".to_string(),
        );
        manager.add_user(user).unwrap();
        assert!(manager.get_user("user1").is_some());
    }

    #[test]
    fn test_effective_permissions() {
        let mut manager = RoleManager::new();

        let mut user = User::new(
            "user1".to_string(),
            "testuser".to_string(),
            "test@example.com".to_string(),
            "Test User".to_string(),
        );
        user.add_role("viewer".to_string());

        let effective_permissions = manager.compute_effective_permissions(&user);
        assert!(effective_permissions.has_permission(&Permission::ViewExplanations));
        assert!(!effective_permissions.has_permission(&Permission::GenerateExplanations));
    }

    #[test]
    fn test_access_control() {
        let config = AccessControlConfig::default();
        let mut ac = AccessControl::new(config);

        // Add a user
        let mut user = User::new(
            "user1".to_string(),
            "testuser".to_string(),
            "test@example.com".to_string(),
            "Test User".to_string(),
        );
        user.add_role("viewer".to_string());
        ac.role_manager_mut().add_user(user).unwrap();

        // Create session
        let context = ac.create_session("user1", "session1".to_string()).unwrap();
        assert_eq!(context.user.id, "user1");
        assert!(context.has_permission(&Permission::ViewExplanations));
    }

    #[test]
    fn test_security_context() {
        let manager = RoleManager::new();
        let mut user = User::new(
            "user1".to_string(),
            "testuser".to_string(),
            "test@example.com".to_string(),
            "Test User".to_string(),
        );
        user.add_role("analyst".to_string());

        let context = SecurityContext::for_user(user, &manager, "session1".to_string());

        assert!(context
            .require_permission(&Permission::ViewExplanations)
            .is_ok());
        assert!(context
            .require_permission(&Permission::GenerateExplanations)
            .is_ok());
        assert!(context
            .require_permission(&Permission::DeleteUsers)
            .is_err());
    }
}
