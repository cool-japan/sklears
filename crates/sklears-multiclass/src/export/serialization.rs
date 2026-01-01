//! Model Serialization
//!
//! Provides JSON and binary serialization for models.

#[cfg(feature = "serde")]
use sklears_core::error::Result as SklResult;
#[cfg(feature = "serde")]
use sklears_core::prelude::SklearsError;
#[cfg(feature = "serde")]
use std::path::Path;

/// Serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// JSON format
    JSON,
    /// MessagePack (binary)
    MessagePack,
    /// Bincode (binary)
    Bincode,
}

/// Serialize model to JSON string
#[cfg(feature = "serde")]
pub fn to_json<T: serde::Serialize>(model: &T) -> SklResult<String> {
    serde_json::to_string_pretty(model)
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to serialize to JSON: {}", e)))
}

/// Deserialize model from JSON string
#[cfg(feature = "serde")]
pub fn from_json<T: serde::de::DeserializeOwned>(json: &str) -> SklResult<T> {
    serde_json::from_str(json)
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to deserialize from JSON: {}", e)))
}

/// Serialize model to bytes
#[cfg(feature = "serde")]
pub fn to_bytes<T: serde::Serialize>(model: &T, format: SerializationFormat) -> SklResult<Vec<u8>> {
    match format {
        SerializationFormat::JSON => {
            let json = to_json(model)?;
            Ok(json.into_bytes())
        }
        SerializationFormat::MessagePack => rmp_serde::to_vec(model).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to serialize to MessagePack: {}", e))
        }),
        SerializationFormat::Bincode => {
            oxicode::serde::encode_to_vec(model, oxicode::config::standard()).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to serialize to Bincode: {}", e))
            })
        }
    }
}

/// Save model to file
#[cfg(feature = "serde")]
pub fn save_to_file<T: serde::Serialize>(
    model: &T,
    path: &Path,
    format: SerializationFormat,
) -> SklResult<()> {
    let bytes = to_bytes(model, format)?;
    std::fs::write(path, bytes)
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to write file: {}", e)))
}

/// Load model from file
#[cfg(feature = "serde")]
pub fn load_from_file<T: serde::de::DeserializeOwned>(
    path: &Path,
    format: SerializationFormat,
) -> SklResult<T> {
    let bytes = std::fs::read(path)
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to read file: {}", e)))?;

    from_bytes(&bytes, format)
}

/// Deserialize model from bytes
#[cfg(feature = "serde")]
pub fn from_bytes<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    format: SerializationFormat,
) -> SklResult<T> {
    match format {
        SerializationFormat::JSON => {
            let json = std::str::from_utf8(bytes)
                .map_err(|e| SklearsError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;
            from_json(json)
        }
        SerializationFormat::MessagePack => rmp_serde::from_slice(bytes).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to deserialize from MessagePack: {}", e))
        }),
        SerializationFormat::Bincode => {
            let (value, _bytes_read) = oxicode::serde::decode_from_slice(
                bytes,
                oxicode::config::standard(),
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to deserialize from Bincode: {}", e))
            })?;
            Ok(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "serde")]
    fn test_json_serialization() {
        use std::collections::HashMap;

        let mut data = HashMap::new();
        data.insert("key", "value");

        let json = to_json(&data).unwrap();
        assert!(json.contains("key"));
        assert!(json.contains("value"));
    }

    #[test]
    fn test_serialization_format() {
        assert_eq!(SerializationFormat::JSON, SerializationFormat::JSON);
        assert_ne!(SerializationFormat::JSON, SerializationFormat::Bincode);
    }
}
