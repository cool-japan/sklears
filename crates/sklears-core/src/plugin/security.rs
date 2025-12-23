//! Plugin Security Framework
//!
//! This module provides comprehensive security infrastructure for the plugin system,
//! including permission management, digital signatures, trust policies, and
//! security validation frameworks.

use crate::error::Result;
use std::collections::HashMap;

/// Security policy configuration for the plugin system
///
/// The SecurityPolicy defines the security requirements and restrictions
/// for plugin validation and execution. It provides configurable security
/// levels to balance functionality with safety requirements.
///
/// # Security Levels
///
/// - **Strict**: Maximum security with signature requirements and minimal permissions
/// - **Standard**: Balanced security for production environments
/// - **Permissive**: Relaxed security for development and testing
///
/// # Examples
///
/// ```rust
/// use sklears_core::plugin::{SecurityPolicy, Permission};
///
/// // Create a strict security policy
/// let strict_policy = SecurityPolicy::strict();
/// assert!(strict_policy.require_signatures);
/// assert!(!strict_policy.allow_unsafe_code);
///
/// // Create a custom policy
/// let custom_policy = SecurityPolicy {
///     allow_unsafe_code: false,
///     require_signatures: true,
///     min_trust_level: 7,
///     max_complexity: 15,
///     dangerous_permissions: vec![
///         "file_system_write".to_string(),
///         "network_access".to_string(),
///     ],
///     restricted_apis: vec![
///         "std::process::Command".to_string(),
///     ],
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Whether to allow plugins with unsafe code blocks
    pub allow_unsafe_code: bool,
    /// Whether digital signatures are required for plugins
    pub require_signatures: bool,
    /// Minimum trust level required for plugin publishers (0-10)
    pub min_trust_level: u8,
    /// Maximum allowed cyclomatic complexity for plugin code
    pub max_complexity: u32,
    /// List of dangerous permissions that trigger warnings
    pub dangerous_permissions: Vec<String>,
    /// List of restricted API patterns that are not allowed
    pub restricted_apis: Vec<String>,
}

impl SecurityPolicy {
    /// Create a strict security policy suitable for production environments
    ///
    /// This policy provides maximum security with signature requirements,
    /// no unsafe code, and minimal dangerous permissions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::SecurityPolicy;
    ///
    /// let policy = SecurityPolicy::strict();
    /// assert!(policy.require_signatures);
    /// assert!(!policy.allow_unsafe_code);
    /// assert_eq!(policy.min_trust_level, 8);
    /// ```
    pub fn strict() -> Self {
        Self {
            allow_unsafe_code: false,
            require_signatures: true,
            min_trust_level: 8,
            max_complexity: 10,
            dangerous_permissions: vec![
                "file_system_write".to_string(),
                "file_system_delete".to_string(),
                "network_access".to_string(),
                "system_commands".to_string(),
                "environment_variables".to_string(),
                "process_spawn".to_string(),
            ],
            restricted_apis: vec![
                "std::process::Command".to_string(),
                "std::fs::remove_file".to_string(),
                "std::fs::remove_dir".to_string(),
                "std::env::set_var".to_string(),
                "libc::system".to_string(),
            ],
        }
    }

    /// Create a standard security policy for typical production use
    ///
    /// Balances security with functionality, allowing more flexibility
    /// than strict mode while maintaining essential protections.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::SecurityPolicy;
    ///
    /// let policy = SecurityPolicy::standard();
    /// assert!(policy.require_signatures);
    /// assert!(!policy.allow_unsafe_code);
    /// assert_eq!(policy.min_trust_level, 5);
    /// ```
    pub fn standard() -> Self {
        Self {
            allow_unsafe_code: false,
            require_signatures: true,
            min_trust_level: 5,
            max_complexity: 20,
            dangerous_permissions: vec![
                "file_system_write".to_string(),
                "network_access".to_string(),
                "system_commands".to_string(),
            ],
            restricted_apis: vec![
                "std::process::Command".to_string(),
                "std::fs::remove_file".to_string(),
            ],
        }
    }

    /// Create a permissive security policy for development and testing
    ///
    /// Provides minimal security restrictions to enable rapid development
    /// and testing. Should not be used in production environments.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::SecurityPolicy;
    ///
    /// let policy = SecurityPolicy::permissive();
    /// assert!(!policy.require_signatures);
    /// assert!(policy.allow_unsafe_code);
    /// assert_eq!(policy.min_trust_level, 0);
    /// ```
    pub fn permissive() -> Self {
        Self {
            allow_unsafe_code: true,
            require_signatures: false,
            min_trust_level: 0,
            max_complexity: 100,
            dangerous_permissions: vec!["system_commands".to_string()],
            restricted_apis: vec![],
        }
    }

    /// Check if a permission is considered dangerous
    ///
    /// Dangerous permissions are those that could potentially be used
    /// maliciously or cause system damage if misused.
    ///
    /// # Arguments
    ///
    /// * `permission` - The permission to check
    ///
    /// # Returns
    ///
    /// true if the permission is considered dangerous, false otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::{SecurityPolicy, Permission};
    ///
    /// let policy = SecurityPolicy::standard();
    /// assert!(policy.is_dangerous_permission(&Permission::FileSystemWrite));
    /// assert!(!policy.is_dangerous_permission(&Permission::FileSystemRead));
    /// ```
    pub fn is_dangerous_permission(&self, permission: &Permission) -> bool {
        match permission {
            Permission::FileSystemWrite => true,
            Permission::FileSystemDelete => true,
            Permission::NetworkAccess => self
                .dangerous_permissions
                .contains(&"network_access".to_string()),
            Permission::SystemCommands => true,
            Permission::ProcessSpawn => true,
            Permission::EnvironmentVariables => self
                .dangerous_permissions
                .contains(&"environment_variables".to_string()),
            Permission::Custom(name) => self.dangerous_permissions.contains(name),
            _ => false,
        }
    }

    /// Check if an API call is restricted
    ///
    /// Restricted APIs are those that are prohibited from use in plugins
    /// due to security concerns or system stability issues.
    ///
    /// # Arguments
    ///
    /// * `api` - The API call pattern to check
    ///
    /// # Returns
    ///
    /// true if the API is restricted, false otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::SecurityPolicy;
    ///
    /// let policy = SecurityPolicy::standard();
    /// assert!(policy.is_restricted_api("std::process::Command"));
    /// assert!(!policy.is_restricted_api("std::fs::read_to_string"));
    /// ```
    pub fn is_restricted_api(&self, api: &str) -> bool {
        self.restricted_apis
            .iter()
            .any(|restricted| api.contains(restricted) || api.starts_with(restricted))
    }

    /// Add a dangerous permission to the policy
    ///
    /// # Arguments
    ///
    /// * `permission` - The permission name to add as dangerous
    pub fn add_dangerous_permission(&mut self, permission: String) {
        if !self.dangerous_permissions.contains(&permission) {
            self.dangerous_permissions.push(permission);
        }
    }

    /// Remove a dangerous permission from the policy
    ///
    /// # Arguments
    ///
    /// * `permission` - The permission name to remove from dangerous list
    pub fn remove_dangerous_permission(&mut self, permission: &str) {
        self.dangerous_permissions.retain(|p| p != permission);
    }

    /// Add a restricted API pattern to the policy
    ///
    /// # Arguments
    ///
    /// * `api_pattern` - The API pattern to restrict
    pub fn add_restricted_api(&mut self, api_pattern: String) {
        if !self.restricted_apis.contains(&api_pattern) {
            self.restricted_apis.push(api_pattern);
        }
    }

    /// Remove a restricted API pattern from the policy
    ///
    /// # Arguments
    ///
    /// * `api_pattern` - The API pattern to remove from restrictions
    pub fn remove_restricted_api(&mut self, api_pattern: &str) {
        self.restricted_apis.retain(|p| p != api_pattern);
    }

    /// Validate the policy configuration
    ///
    /// Ensures that the policy configuration is consistent and valid.
    ///
    /// # Returns
    ///
    /// Ok(()) if the policy is valid, error otherwise.
    pub fn validate(&self) -> Result<()> {
        if self.min_trust_level > 10 {
            return Err(crate::error::SklearsError::InvalidOperation(
                "Trust level must be between 0 and 10".to_string(),
            ));
        }

        if self.max_complexity == 0 {
            return Err(crate::error::SklearsError::InvalidOperation(
                "Maximum complexity must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

/// Plugin permission types
///
/// Permissions define what system resources and capabilities a plugin
/// requires to function. They are used for security validation and
/// user consent workflows.
///
/// # Permission Categories
///
/// - **File System**: Read, write, and delete file operations
/// - **Network**: Internet and local network access
/// - **System**: Process execution and system command access
/// - **Hardware**: GPU and specialized hardware access
/// - **Environment**: Access to environment variables and system settings
///
/// # Examples
///
/// ```rust
/// use sklears_core::plugin::Permission;
///
/// // Standard permissions
/// let read_perm = Permission::FileSystemRead;
/// let write_perm = Permission::FileSystemWrite;
/// let network_perm = Permission::NetworkAccess;
///
/// // Custom permission
/// let custom_perm = Permission::Custom("database_access".to_string());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Permission {
    /// Read access to file system
    FileSystemRead,
    /// Write access to file system
    FileSystemWrite,
    /// Delete access to file system
    FileSystemDelete,
    /// Access to network resources
    NetworkAccess,
    /// Execute system commands
    SystemCommands,
    /// Spawn new processes
    ProcessSpawn,
    /// Access to GPU resources
    GpuAccess,
    /// Access to environment variables
    EnvironmentVariables,
    /// Access to system configuration
    SystemConfiguration,
    /// Custom permission with user-defined name
    Custom(String),
}

impl Permission {
    /// Get a human-readable description of the permission
    ///
    /// # Returns
    ///
    /// A string describing what this permission allows.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::Permission;
    ///
    /// let perm = Permission::FileSystemWrite;
    /// assert_eq!(perm.description(), "Write access to file system");
    /// ```
    pub fn description(&self) -> &'static str {
        match self {
            Permission::FileSystemRead => "Read access to file system",
            Permission::FileSystemWrite => "Write access to file system",
            Permission::FileSystemDelete => "Delete access to file system",
            Permission::NetworkAccess => "Access to network resources",
            Permission::SystemCommands => "Execute system commands",
            Permission::ProcessSpawn => "Spawn new processes",
            Permission::GpuAccess => "Access to GPU resources",
            Permission::EnvironmentVariables => "Access to environment variables",
            Permission::SystemConfiguration => "Access to system configuration",
            Permission::Custom(_) => "Custom permission",
        }
    }

    /// Get the risk level of this permission
    ///
    /// # Returns
    ///
    /// Risk level from 1 (low) to 5 (high).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::Permission;
    ///
    /// assert_eq!(Permission::FileSystemRead.risk_level(), 2);
    /// assert_eq!(Permission::SystemCommands.risk_level(), 5);
    /// ```
    pub fn risk_level(&self) -> u8 {
        match self {
            Permission::FileSystemRead => 2,
            Permission::GpuAccess => 2,
            Permission::FileSystemWrite => 3,
            Permission::NetworkAccess => 3,
            Permission::EnvironmentVariables => 3,
            Permission::FileSystemDelete => 4,
            Permission::SystemConfiguration => 4,
            Permission::ProcessSpawn => 5,
            Permission::SystemCommands => 5,
            Permission::Custom(_) => 3, // Default to medium risk
        }
    }

    /// Check if this permission requires user consent
    ///
    /// # Returns
    ///
    /// true if user consent is required, false otherwise.
    pub fn requires_user_consent(&self) -> bool {
        self.risk_level() >= 3
    }

    /// Get all standard permissions
    ///
    /// # Returns
    ///
    /// A vector of all predefined permission types.
    pub fn all_standard() -> Vec<Permission> {
        vec![
            Permission::FileSystemRead,
            Permission::FileSystemWrite,
            Permission::FileSystemDelete,
            Permission::NetworkAccess,
            Permission::SystemCommands,
            Permission::ProcessSpawn,
            Permission::GpuAccess,
            Permission::EnvironmentVariables,
            Permission::SystemConfiguration,
        ]
    }
}

/// Digital signature for plugin verification
///
/// Digital signatures provide cryptographic verification of plugin integrity
/// and authenticity. They ensure that plugins have not been tampered with
/// and come from trusted sources.
///
/// # Supported Algorithms
///
/// - RSA-SHA256: Standard RSA signature with SHA-256 hashing
/// - ECDSA-P256: Elliptic curve signature with P-256 curve
/// - Ed25519: Edwards curve signature algorithm
///
/// # Examples
///
/// ```rust
/// use sklears_core::plugin::DigitalSignature;
///
/// let signature = DigitalSignature {
///     algorithm: "RSA-SHA256".to_string(),
///     signature: vec![0x12, 0x34, 0x56, 0x78],
///     public_key_fingerprint: "SHA256:abc123def456".to_string(),
///     timestamp: std::time::SystemTime::now(),
///     signer_certificate: Some("-----BEGIN CERTIFICATE-----\n...".to_string()),
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DigitalSignature {
    /// Signature algorithm used (e.g., "RSA-SHA256", "ECDSA-P256", "Ed25519")
    pub algorithm: String,
    /// The actual signature bytes
    pub signature: Vec<u8>,
    /// Fingerprint of the public key used for verification
    pub public_key_fingerprint: String,
    /// Timestamp when the signature was created
    pub timestamp: std::time::SystemTime,
    /// Optional signer certificate in PEM format
    pub signer_certificate: Option<String>,
}

impl DigitalSignature {
    /// Create a new digital signature
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The signature algorithm used
    /// * `signature` - The signature bytes
    /// * `public_key_fingerprint` - Fingerprint of the signing key
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::DigitalSignature;
    ///
    /// let sig = DigitalSignature::new(
    ///     "RSA-SHA256".to_string(),
    ///     vec![0x12, 0x34],
    ///     "SHA256:abc123".to_string(),
    /// );
    /// ```
    pub fn new(algorithm: String, signature: Vec<u8>, public_key_fingerprint: String) -> Self {
        Self {
            algorithm,
            signature,
            public_key_fingerprint,
            timestamp: std::time::SystemTime::now(),
            signer_certificate: None,
        }
    }

    /// Verify the signature algorithm is supported
    ///
    /// # Returns
    ///
    /// true if the algorithm is supported, false otherwise.
    pub fn is_algorithm_supported(&self) -> bool {
        matches!(
            self.algorithm.as_str(),
            "RSA-SHA256" | "RSA-SHA512" | "ECDSA-P256" | "ECDSA-P384" | "Ed25519"
        )
    }

    /// Get the security strength of the signature algorithm
    ///
    /// # Returns
    ///
    /// Security strength in bits, or 0 for unknown algorithms.
    pub fn security_strength(&self) -> u32 {
        match self.algorithm.as_str() {
            "RSA-SHA256" => 112, // 2048-bit RSA
            "RSA-SHA512" => 112, // 2048-bit RSA
            "ECDSA-P256" => 128, // P-256 curve
            "ECDSA-P384" => 192, // P-384 curve
            "Ed25519" => 128,    // Ed25519 curve
            _ => 0,              // Unknown algorithm
        }
    }

    /// Check if the signature has expired
    ///
    /// # Arguments
    ///
    /// * `max_age_seconds` - Maximum age in seconds before expiration
    ///
    /// # Returns
    ///
    /// true if the signature has expired, false otherwise.
    pub fn is_expired(&self, max_age_seconds: u64) -> bool {
        if let Ok(elapsed) = self.timestamp.elapsed() {
            elapsed.as_secs() > max_age_seconds
        } else {
            true // If we can't determine age, consider it expired
        }
    }
}

/// Trust store for managing trusted keys and certificates
///
/// The TrustStore manages the cryptographic keys and certificates used
/// for plugin signature verification. It provides a secure storage
/// mechanism for trusted publisher keys.
///
/// # Examples
///
/// ```rust
/// use sklears_core::plugin::TrustStore;
///
/// let trust_store = TrustStore::new();
///
/// // In a real implementation, you would load keys from secure storage
/// // trust_store.load_from_file("trust_store.pem")?;
/// ```
#[derive(Debug, Clone)]
pub struct TrustStore {
    /// Trusted public keys indexed by fingerprint
    trusted_keys: HashMap<String, PublicKeyInfo>,
    /// Trusted certificates indexed by subject
    trusted_certificates: HashMap<String, CertificateInfo>,
    /// Revoked key fingerprints
    revoked_keys: Vec<String>,
}

impl TrustStore {
    /// Create a new empty trust store
    pub fn new() -> Self {
        Self {
            trusted_keys: HashMap::new(),
            trusted_certificates: HashMap::new(),
            revoked_keys: Vec::new(),
        }
    }

    /// Verify a digital signature against the trust store
    ///
    /// # Arguments
    ///
    /// * `content_hash` - Hash of the content being verified
    /// * `signature` - The digital signature to verify
    ///
    /// # Returns
    ///
    /// Ok(true) if signature is valid, Ok(false) if invalid, Err if verification fails.
    pub fn verify_signature(
        &self,
        _content_hash: &str,
        signature: &DigitalSignature,
    ) -> Result<bool> {
        // Check if the signing key is revoked
        if self
            .revoked_keys
            .contains(&signature.public_key_fingerprint)
        {
            return Ok(false);
        }

        // Check if algorithm is supported
        if !signature.is_algorithm_supported() {
            return Err(crate::error::SklearsError::InvalidOperation(format!(
                "Unsupported signature algorithm: {}",
                signature.algorithm
            )));
        }

        // Check if signature has expired (1 year default)
        if signature.is_expired(365 * 24 * 60 * 60) {
            return Ok(false);
        }

        // Look up the public key
        if let Some(key_info) = self.trusted_keys.get(&signature.public_key_fingerprint) {
            if key_info.is_valid() {
                // In a real implementation, this would perform actual cryptographic verification
                // For now, we simulate verification based on fingerprint match
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            // Key not found in trust store
            Ok(false)
        }
    }

    /// Get the trust level of a publisher
    ///
    /// # Arguments
    ///
    /// * `publisher` - The publisher information to evaluate
    ///
    /// # Returns
    ///
    /// Trust level from 0 (untrusted) to 10 (fully trusted).
    pub fn get_publisher_trust(&self, publisher: &PublisherInfo) -> u8 {
        let mut trust_level = 0u8;

        // Base trust for verified publishers
        if publisher.verified {
            trust_level += 3;
        }

        // Additional trust based on publisher trust score
        trust_level += (publisher.trust_score / 2).min(5);

        // Bonus for having certificates in trust store
        if self.trusted_certificates.contains_key(&publisher.name) {
            trust_level += 2;
        }

        trust_level.min(10)
    }

    /// Add a trusted public key to the store
    ///
    /// # Arguments
    ///
    /// * `fingerprint` - The key fingerprint
    /// * `key_info` - Information about the public key
    pub fn add_trusted_key(&mut self, fingerprint: String, key_info: PublicKeyInfo) {
        self.trusted_keys.insert(fingerprint, key_info);
    }

    /// Remove a trusted key from the store
    ///
    /// # Arguments
    ///
    /// * `fingerprint` - The key fingerprint to remove
    pub fn remove_trusted_key(&mut self, fingerprint: &str) {
        self.trusted_keys.remove(fingerprint);
    }

    /// Revoke a key by adding it to the revocation list
    ///
    /// # Arguments
    ///
    /// * `fingerprint` - The key fingerprint to revoke
    pub fn revoke_key(&mut self, fingerprint: String) {
        if !self.revoked_keys.contains(&fingerprint) {
            self.revoked_keys.push(fingerprint.clone());
        }
        // Also remove from trusted keys if present
        self.trusted_keys.remove(&fingerprint);
    }

    /// Check if a key is revoked
    ///
    /// # Arguments
    ///
    /// * `fingerprint` - The key fingerprint to check
    ///
    /// # Returns
    ///
    /// true if the key is revoked, false otherwise.
    pub fn is_key_revoked(&self, fingerprint: &str) -> bool {
        self.revoked_keys.contains(&fingerprint.to_string())
    }

    /// Load trust store from a file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the trust store file
    ///
    /// # Returns
    ///
    /// Ok(()) on success, error on failure.
    pub fn load_from_file(&mut self, _path: &str) -> Result<()> {
        // Placeholder implementation
        // In practice, this would load keys and certificates from a file
        Ok(())
    }

    /// Save trust store to a file
    ///
    /// # Arguments
    ///
    /// * `path` - Path where to save the trust store
    ///
    /// # Returns
    ///
    /// Ok(()) on success, error on failure.
    pub fn save_to_file(&self, _path: &str) -> Result<()> {
        // Placeholder implementation
        // In practice, this would save keys and certificates to a file
        Ok(())
    }
}

impl Default for TrustStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a publisher
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PublisherInfo {
    /// Publisher name
    pub name: String,
    /// Publisher email
    pub email: String,
    /// Publisher website
    pub website: Option<String>,
    /// Whether the publisher is verified
    pub verified: bool,
    /// Trust score (0-10)
    pub trust_score: u8,
}

impl PublisherInfo {
    /// Create a new publisher info
    pub fn new(name: String, email: String) -> Self {
        Self {
            name,
            email,
            website: None,
            verified: false,
            trust_score: 0,
        }
    }

    /// Check if the publisher information is complete
    pub fn is_complete(&self) -> bool {
        !self.name.is_empty() && !self.email.is_empty() && self.email.contains('@')
    }
}

/// Information about a public key in the trust store
#[derive(Debug, Clone)]
pub struct PublicKeyInfo {
    /// Key algorithm (e.g., "RSA", "ECDSA", "Ed25519")
    pub algorithm: String,
    /// Key size in bits
    pub key_size: u32,
    /// When the key was added to the trust store
    pub added_timestamp: std::time::SystemTime,
    /// Optional expiration time for the key
    pub expires_at: Option<std::time::SystemTime>,
    /// Owner of the key
    pub owner: String,
}

impl PublicKeyInfo {
    /// Check if the key is still valid (not expired)
    pub fn is_valid(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            std::time::SystemTime::now() < expires_at
        } else {
            true // No expiration set
        }
    }

    /// Get the security strength of this key
    pub fn security_strength(&self) -> u32 {
        match self.algorithm.as_str() {
            "RSA" => {
                if self.key_size >= 2048 {
                    112
                } else if self.key_size >= 1024 {
                    80
                } else {
                    0 // Too weak
                }
            }
            "ECDSA" => {
                if self.key_size >= 256 {
                    128
                } else {
                    80
                }
            }
            "Ed25519" => 128,
            _ => 0,
        }
    }
}

/// Information about a certificate in the trust store
#[derive(Debug, Clone)]
pub struct CertificateInfo {
    /// Certificate subject
    pub subject: String,
    /// Certificate issuer
    pub issuer: String,
    /// Certificate validity period
    pub not_before: std::time::SystemTime,
    /// Certificate expiration
    pub not_after: std::time::SystemTime,
    /// Certificate fingerprint
    pub fingerprint: String,
}

impl CertificateInfo {
    /// Check if the certificate is currently valid
    pub fn is_valid(&self) -> bool {
        let now = std::time::SystemTime::now();
        now >= self.not_before && now <= self.not_after
    }
}

/// Permission set for grouping related permissions
#[derive(Debug, Clone)]
pub struct PermissionSet {
    /// Name of the permission set
    pub name: String,
    /// Description of what this set enables
    pub description: String,
    /// Individual permissions in this set
    pub permissions: Vec<Permission>,
}

impl PermissionSet {
    /// Create a new permission set
    pub fn new(name: String, description: String, permissions: Vec<Permission>) -> Self {
        Self {
            name,
            description,
            permissions,
        }
    }

    /// Create a file system permission set
    pub fn file_system() -> Self {
        Self::new(
            "file_system".to_string(),
            "Read and write access to the file system".to_string(),
            vec![Permission::FileSystemRead, Permission::FileSystemWrite],
        )
    }

    /// Create a network permission set
    pub fn network() -> Self {
        Self::new(
            "network".to_string(),
            "Access to network resources".to_string(),
            vec![Permission::NetworkAccess],
        )
    }

    /// Create a system administration permission set
    pub fn system_admin() -> Self {
        Self::new(
            "system_admin".to_string(),
            "Administrative access to system resources".to_string(),
            vec![
                Permission::SystemCommands,
                Permission::ProcessSpawn,
                Permission::EnvironmentVariables,
                Permission::SystemConfiguration,
            ],
        )
    }

    /// Get the maximum risk level in this permission set
    pub fn max_risk_level(&self) -> u8 {
        self.permissions
            .iter()
            .map(|p| p.risk_level())
            .max()
            .unwrap_or(0)
    }

    /// Check if any permissions require user consent
    pub fn requires_user_consent(&self) -> bool {
        self.permissions.iter().any(|p| p.requires_user_consent())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_policy_levels() {
        let strict = SecurityPolicy::strict();
        assert!(strict.require_signatures);
        assert!(!strict.allow_unsafe_code);
        assert_eq!(strict.min_trust_level, 8);

        let standard = SecurityPolicy::standard();
        assert!(standard.require_signatures);
        assert!(!standard.allow_unsafe_code);
        assert_eq!(standard.min_trust_level, 5);

        let permissive = SecurityPolicy::permissive();
        assert!(!permissive.require_signatures);
        assert!(permissive.allow_unsafe_code);
        assert_eq!(permissive.min_trust_level, 0);
    }

    #[test]
    fn test_permission_risk_levels() {
        assert_eq!(Permission::FileSystemRead.risk_level(), 2);
        assert_eq!(Permission::FileSystemWrite.risk_level(), 3);
        assert_eq!(Permission::SystemCommands.risk_level(), 5);
        assert_eq!(Permission::GpuAccess.risk_level(), 2);
    }

    #[test]
    fn test_permission_consent_requirements() {
        assert!(!Permission::FileSystemRead.requires_user_consent());
        assert!(Permission::FileSystemWrite.requires_user_consent());
        assert!(Permission::SystemCommands.requires_user_consent());
        assert!(!Permission::GpuAccess.requires_user_consent());
    }

    #[test]
    fn test_digital_signature_algorithms() {
        let rsa_sig = DigitalSignature::new("RSA-SHA256".to_string(), vec![], "test".to_string());
        assert!(rsa_sig.is_algorithm_supported());
        assert_eq!(rsa_sig.security_strength(), 112);

        let unknown_sig = DigitalSignature::new("UNKNOWN".to_string(), vec![], "test".to_string());
        assert!(!unknown_sig.is_algorithm_supported());
        assert_eq!(unknown_sig.security_strength(), 0);
    }

    #[test]
    fn test_trust_store_key_management() {
        let mut trust_store = TrustStore::new();

        let key_info = PublicKeyInfo {
            algorithm: "RSA".to_string(),
            key_size: 2048,
            added_timestamp: std::time::SystemTime::now(),
            expires_at: None,
            owner: "test@example.com".to_string(),
        };

        // Add trusted key
        trust_store.add_trusted_key("fingerprint123".to_string(), key_info.clone());
        assert!(!trust_store.is_key_revoked("fingerprint123"));

        // Revoke key
        trust_store.revoke_key("fingerprint123".to_string());
        assert!(trust_store.is_key_revoked("fingerprint123"));
    }

    #[test]
    fn test_permission_sets() {
        let fs_set = PermissionSet::file_system();
        assert_eq!(fs_set.permissions.len(), 2);
        assert!(fs_set.requires_user_consent());
        assert_eq!(fs_set.max_risk_level(), 3);

        let admin_set = PermissionSet::system_admin();
        assert!(admin_set.permissions.len() > 2);
        assert!(admin_set.requires_user_consent());
        assert_eq!(admin_set.max_risk_level(), 5);
    }

    #[test]
    fn test_publisher_info() {
        let publisher =
            PublisherInfo::new("Test Publisher".to_string(), "test@example.com".to_string());
        assert!(publisher.is_complete());

        let incomplete = PublisherInfo::new("".to_string(), "invalid-email".to_string());
        assert!(!incomplete.is_complete());
    }
}
