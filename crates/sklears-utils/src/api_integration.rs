//! API integration utilities for ML workflows
//!
//! This module provides HTTP client utilities, REST API helpers, and integration
//! patterns for machine learning services and external APIs.

use crate::UtilsError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// API client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub base_url: String,
    pub timeout: Duration,
    pub max_retries: u32,
    pub retry_delay: Duration,
    pub headers: HashMap<String, String>,
    pub authentication: Option<Authentication>,
    pub user_agent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Authentication {
    Bearer(String),
    ApiKey { key: String, header: String },
    Basic { username: String, password: String },
    Custom { headers: HashMap<String, String> },
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.example.com".to_string(),
            timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_millis(1000),
            headers: HashMap::new(),
            authentication: None,
            user_agent: "sklears-utils/1.0".to_string(),
        }
    }
}

impl ApiConfig {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            ..Default::default()
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_retries(mut self, max_retries: u32, retry_delay: Duration) -> Self {
        self.max_retries = max_retries;
        self.retry_delay = retry_delay;
        self
    }

    pub fn with_authentication(mut self, auth: Authentication) -> Self {
        self.authentication = Some(auth);
        self
    }

    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }

    pub fn with_user_agent(mut self, user_agent: String) -> Self {
        self.user_agent = user_agent;
        self
    }
}

/// API error types
#[derive(thiserror::Error, Debug, Clone)]
pub enum ApiError {
    #[error("HTTP error {status}: {message}")]
    HttpError { status: u16, message: String },
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Timeout error: request took longer than {0:?}")]
    TimeoutError(Duration),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Authentication error: {0}")]
    AuthenticationError(String),
    #[error("Rate limit exceeded: {retry_after:?}")]
    RateLimitError { retry_after: Option<Duration> },
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

impl From<ApiError> for UtilsError {
    fn from(err: ApiError) -> Self {
        UtilsError::InvalidParameter(err.to_string())
    }
}

/// HTTP method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HttpMethod::Get => write!(f, "GET"),
            HttpMethod::Post => write!(f, "POST"),
            HttpMethod::Put => write!(f, "PUT"),
            HttpMethod::Delete => write!(f, "DELETE"),
            HttpMethod::Patch => write!(f, "PATCH"),
            HttpMethod::Head => write!(f, "HEAD"),
            HttpMethod::Options => write!(f, "OPTIONS"),
        }
    }
}

/// HTTP request representation
#[derive(Debug, Clone)]
pub struct ApiRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
    pub query_params: HashMap<String, String>,
}

impl ApiRequest {
    pub fn new(method: HttpMethod, url: String) -> Self {
        Self {
            method,
            url,
            headers: HashMap::new(),
            body: None,
            query_params: HashMap::new(),
        }
    }

    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }

    pub fn with_body<T: Serialize>(mut self, body: &T) -> Result<Self, ApiError> {
        let serialized =
            serde_json::to_vec(body).map_err(|e| ApiError::SerializationError(e.to_string()))?;
        self.body = Some(serialized);
        self.headers
            .insert("Content-Type".to_string(), "application/json".to_string());
        Ok(self)
    }

    pub fn with_json_body(mut self, json: String) -> Self {
        self.body = Some(json.into_bytes());
        self.headers
            .insert("Content-Type".to_string(), "application/json".to_string());
        self
    }

    pub fn with_query_param(mut self, key: String, value: String) -> Self {
        self.query_params.insert(key, value);
        self
    }

    pub fn with_query_params(mut self, params: HashMap<String, String>) -> Self {
        self.query_params.extend(params);
        self
    }

    pub fn build_url(&self) -> String {
        if self.query_params.is_empty() {
            return self.url.clone();
        }

        let query_string: String = self
            .query_params
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");

        if self.url.contains('?') {
            format!("{}&{}", self.url, query_string)
        } else {
            format!("{}?{}", self.url, query_string)
        }
    }
}

/// HTTP response representation
#[derive(Debug, Clone)]
pub struct ApiResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub execution_time: Duration,
}

impl ApiResponse {
    pub fn new(status_code: u16, body: Vec<u8>, execution_time: Duration) -> Self {
        Self {
            status_code,
            headers: HashMap::new(),
            body,
            execution_time,
        }
    }

    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    pub fn is_success(&self) -> bool {
        self.status_code >= 200 && self.status_code < 300
    }

    pub fn text(&self) -> Result<String, ApiError> {
        String::from_utf8(self.body.clone())
            .map_err(|e| ApiError::SerializationError(e.to_string()))
    }

    pub fn json<T: for<'de> Deserialize<'de>>(&self) -> Result<T, ApiError> {
        serde_json::from_slice(&self.body).map_err(|e| ApiError::SerializationError(e.to_string()))
    }

    pub fn get_header(&self, name: &str) -> Option<&String> {
        self.headers.get(name)
    }
}

/// API client trait for making HTTP requests
pub trait ApiClient {
    fn execute(&self, request: ApiRequest) -> Result<ApiResponse, ApiError>;
}

/// Mock API client for testing
pub struct MockApiClient {
    responses: Arc<Mutex<Vec<ApiResponse>>>,
    current_index: Arc<Mutex<usize>>,
}

impl Default for MockApiClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MockApiClient {
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
            current_index: Arc::new(Mutex::new(0)),
        }
    }

    pub fn add_response(&self, response: ApiResponse) {
        if let Ok(mut responses) = self.responses.lock() {
            responses.push(response);
        }
    }

    pub fn reset(&self) {
        if let Ok(mut responses) = self.responses.lock() {
            responses.clear();
        }
        if let Ok(mut index) = self.current_index.lock() {
            *index = 0;
        }
    }
}

impl ApiClient for MockApiClient {
    fn execute(&self, _request: ApiRequest) -> Result<ApiResponse, ApiError> {
        let responses = self
            .responses
            .lock()
            .map_err(|_| ApiError::NetworkError("Failed to lock responses".to_string()))?;

        let mut index = self
            .current_index
            .lock()
            .map_err(|_| ApiError::NetworkError("Failed to lock index".to_string()))?;

        if *index >= responses.len() {
            return Err(ApiError::NetworkError(
                "No more mock responses available".to_string(),
            ));
        }

        let response = responses[*index].clone();
        *index += 1;
        Ok(response)
    }
}

/// Request builder for fluent API construction
pub struct RequestBuilder {
    method: HttpMethod,
    url: String,
    headers: HashMap<String, String>,
    query_params: HashMap<String, String>,
    body: Option<Vec<u8>>,
}

impl RequestBuilder {
    pub fn new(method: HttpMethod, url: String) -> Self {
        Self {
            method,
            url,
            headers: HashMap::new(),
            query_params: HashMap::new(),
            body: None,
        }
    }

    pub fn get(url: String) -> Self {
        Self::new(HttpMethod::Get, url)
    }

    pub fn post(url: String) -> Self {
        Self::new(HttpMethod::Post, url)
    }

    pub fn put(url: String) -> Self {
        Self::new(HttpMethod::Put, url)
    }

    pub fn delete(url: String) -> Self {
        Self::new(HttpMethod::Delete, url)
    }

    pub fn header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }

    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    pub fn query(mut self, key: String, value: String) -> Self {
        self.query_params.insert(key, value);
        self
    }

    pub fn json<T: Serialize + ?Sized>(mut self, body: &T) -> Result<Self, ApiError> {
        let serialized =
            serde_json::to_vec(body).map_err(|e| ApiError::SerializationError(e.to_string()))?;
        self.body = Some(serialized);
        self.headers
            .insert("Content-Type".to_string(), "application/json".to_string());
        Ok(self)
    }

    pub fn text(mut self, body: String) -> Self {
        self.body = Some(body.into_bytes());
        self.headers
            .insert("Content-Type".to_string(), "text/plain".to_string());
        self
    }

    pub fn build(self) -> ApiRequest {
        ApiRequest {
            method: self.method,
            url: self.url,
            headers: self.headers,
            body: self.body,
            query_params: self.query_params,
        }
    }
}

/// High-level API service wrapper
pub struct ApiService {
    client: Box<dyn ApiClient + Send + Sync>,
    config: ApiConfig,
    metrics: Arc<Mutex<ApiMetrics>>,
}

impl ApiService {
    pub fn new(client: Box<dyn ApiClient + Send + Sync>, config: ApiConfig) -> Self {
        Self {
            client,
            config,
            metrics: Arc::new(Mutex::new(ApiMetrics::default())),
        }
    }

    pub fn with_mock() -> Self {
        Self::new(Box::new(MockApiClient::new()), ApiConfig::default())
    }

    pub fn get(&self, endpoint: &str) -> RequestBuilder {
        let url = format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint.trim_start_matches('/')
        );
        RequestBuilder::get(url).headers(self.build_default_headers())
    }

    pub fn post(&self, endpoint: &str) -> RequestBuilder {
        let url = format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint.trim_start_matches('/')
        );
        RequestBuilder::post(url).headers(self.build_default_headers())
    }

    pub fn put(&self, endpoint: &str) -> RequestBuilder {
        let url = format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint.trim_start_matches('/')
        );
        RequestBuilder::put(url).headers(self.build_default_headers())
    }

    pub fn delete(&self, endpoint: &str) -> RequestBuilder {
        let url = format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint.trim_start_matches('/')
        );
        RequestBuilder::delete(url).headers(self.build_default_headers())
    }

    pub fn execute(&self, request: ApiRequest) -> Result<ApiResponse, ApiError> {
        let start_time = Instant::now();

        // Apply authentication
        let request = self.apply_authentication(request)?;

        // Execute request with retry logic
        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                std::thread::sleep(self.config.retry_delay);
            }

            match self.client.execute(request.clone()) {
                Ok(response) => {
                    let execution_time = start_time.elapsed();
                    self.record_metrics(&request, &response, execution_time);
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e);
                    if !self.should_retry(last_error.as_ref().unwrap()) {
                        break;
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }

    pub fn get_metrics(&self) -> Option<ApiMetrics> {
        self.metrics.lock().ok().map(|m| m.clone())
    }

    pub fn reset_metrics(&self) {
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = ApiMetrics::default();
        }
    }

    fn build_default_headers(&self) -> HashMap<String, String> {
        let mut headers = self.config.headers.clone();
        headers.insert("User-Agent".to_string(), self.config.user_agent.clone());
        headers
    }

    fn apply_authentication(&self, mut request: ApiRequest) -> Result<ApiRequest, ApiError> {
        if let Some(auth) = &self.config.authentication {
            match auth {
                Authentication::Bearer(token) => {
                    request =
                        request.with_header("Authorization".to_string(), format!("Bearer {token}"));
                }
                Authentication::ApiKey { key, header } => {
                    request = request.with_header(header.clone(), key.clone());
                }
                Authentication::Basic { username, password } => {
                    let credentials = base64::encode(format!("{username}:{password}"));
                    request = request
                        .with_header("Authorization".to_string(), format!("Basic {credentials}"));
                }
                Authentication::Custom { headers } => {
                    for (key, value) in headers {
                        request = request.with_header(key.clone(), value.clone());
                    }
                }
            }
        }
        Ok(request)
    }

    fn should_retry(&self, error: &ApiError) -> bool {
        match error {
            ApiError::NetworkError(_) => true,
            ApiError::TimeoutError(_) => true,
            ApiError::HttpError { status, .. } => {
                // Retry on server errors (5xx) and rate limiting (429)
                *status >= 500 || *status == 429
            }
            _ => false,
        }
    }

    fn record_metrics(
        &self,
        request: &ApiRequest,
        response: &ApiResponse,
        execution_time: Duration,
    ) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.total_requests += 1;
            if response.is_success() {
                metrics.successful_requests += 1;
            } else {
                metrics.failed_requests += 1;
            }
            metrics.total_execution_time += execution_time;
            metrics.average_response_time = Duration::from_nanos(
                (metrics.total_execution_time.as_nanos() / metrics.total_requests as u128) as u64,
            );

            let method_stats = metrics
                .method_stats
                .entry(request.method)
                .or_insert(MethodStats::default());
            method_stats.requests += 1;
            method_stats.total_time += execution_time;
            method_stats.average_time = Duration::from_nanos(
                (method_stats.total_time.as_nanos() / method_stats.requests as u128) as u64,
            );
        }
    }
}

/// API usage metrics
#[derive(Debug, Clone, Default)]
pub struct ApiMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_execution_time: Duration,
    pub average_response_time: Duration,
    pub method_stats: HashMap<HttpMethod, MethodStats>,
}

#[derive(Debug, Clone, Default)]
pub struct MethodStats {
    pub requests: u64,
    pub total_time: Duration,
    pub average_time: Duration,
}

impl ApiMetrics {
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }
}

impl fmt::Display for ApiMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "API Metrics:")?;
        writeln!(f, "  Total Requests: {}", self.total_requests)?;
        writeln!(f, "  Success Rate: {:.2}%", self.success_rate() * 100.0)?;
        writeln!(
            f,
            "  Average Response Time: {:?}",
            self.average_response_time
        )?;

        if !self.method_stats.is_empty() {
            writeln!(f, "  Method Statistics:")?;
            for (method, stats) in &self.method_stats {
                writeln!(
                    f,
                    "    {}: {} requests, avg {:?}",
                    method, stats.requests, stats.average_time
                )?;
            }
        }

        Ok(())
    }
}

/// Common ML API patterns
pub struct MLApiPatterns;

impl MLApiPatterns {
    /// Create a prediction request pattern
    pub fn prediction_request<T: Serialize>(
        service: &ApiService,
        endpoint: &str,
        features: &T,
    ) -> Result<RequestBuilder, ApiError> {
        service.post(endpoint).json(features)
    }

    /// Create a batch prediction request pattern
    pub fn batch_prediction_request<T: Serialize>(
        service: &ApiService,
        endpoint: &str,
        batch_features: &[T],
    ) -> Result<RequestBuilder, ApiError> {
        service.post(endpoint).json(batch_features)
    }

    /// Create a model training request pattern
    pub fn training_request<T: Serialize>(
        service: &ApiService,
        endpoint: &str,
        training_data: &T,
        model_config: &HashMap<String, serde_json::Value>,
    ) -> Result<RequestBuilder, ApiError> {
        let payload = serde_json::json!({
            "data": training_data,
            "config": model_config
        });
        service.post(endpoint).json(&payload)
    }

    /// Create a model status check pattern
    pub fn model_status_request(service: &ApiService, model_id: &str) -> RequestBuilder {
        service.get(&format!("models/{model_id}/status"))
    }

    /// Create a health check pattern
    pub fn health_check_request(service: &ApiService) -> RequestBuilder {
        service.get("health")
    }
}

// Simple base64 encoding for basic auth (mock implementation)
mod base64 {
    pub fn encode(input: String) -> String {
        // This is a simplified mock implementation
        // In a real implementation, you would use a proper base64 library
        format!("base64({input})")
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_api_config() {
        let config = ApiConfig::new("https://api.example.com".to_string())
            .with_timeout(Duration::from_secs(60))
            .with_retries(5, Duration::from_millis(500))
            .with_header("Custom-Header".to_string(), "value".to_string());

        assert_eq!(config.base_url, "https://api.example.com");
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_retries, 5);
        assert_eq!(
            config.headers.get("Custom-Header"),
            Some(&"value".to_string())
        );
    }

    #[test]
    fn test_request_builder() {
        let request = RequestBuilder::get("https://api.example.com/test".to_string())
            .header("Content-Type".to_string(), "application/json".to_string())
            .query("param".to_string(), "value".to_string())
            .build();

        assert_eq!(request.method, HttpMethod::Get);
        assert_eq!(request.url, "https://api.example.com/test");
        assert_eq!(
            request.headers.get("Content-Type"),
            Some(&"application/json".to_string())
        );
        assert_eq!(
            request.query_params.get("param"),
            Some(&"value".to_string())
        );
        assert_eq!(
            request.build_url(),
            "https://api.example.com/test?param=value"
        );
    }

    #[test]
    fn test_request_with_json_body() {
        let data = json!({"name": "test", "value": 42});
        let request = RequestBuilder::post("https://api.example.com/data".to_string())
            .json(&data)
            .unwrap()
            .build();

        assert_eq!(request.method, HttpMethod::Post);
        assert!(request.body.is_some());
        assert_eq!(
            request.headers.get("Content-Type"),
            Some(&"application/json".to_string())
        );
    }

    #[test]
    fn test_api_response() {
        let body = b"{\"result\": \"success\"}".to_vec();
        let response = ApiResponse::new(200, body, Duration::from_millis(100));

        assert!(response.is_success());
        assert_eq!(response.status_code, 200);
        assert_eq!(response.execution_time, Duration::from_millis(100));

        let text = response.text().unwrap();
        assert_eq!(text, "{\"result\": \"success\"}");

        let json: serde_json::Value = response.json().unwrap();
        assert_eq!(json["result"], "success");
    }

    #[test]
    fn test_mock_api_client() {
        let client = MockApiClient::new();

        let mock_response = ApiResponse::new(
            200,
            b"{\"data\": \"test\"}".to_vec(),
            Duration::from_millis(50),
        );
        client.add_response(mock_response);

        let request = ApiRequest::new(HttpMethod::Get, "https://api.example.com/test".to_string());
        let response = client.execute(request).unwrap();

        assert_eq!(response.status_code, 200);
        assert_eq!(response.text().unwrap(), "{\"data\": \"test\"}");
    }

    #[test]
    fn test_api_service() {
        let service = ApiService::with_mock();

        // Get the mock client and add a response
        let request_builder = service.get("test");
        let request = request_builder.build();

        // This would normally fail since we haven't added any mock responses
        // but it tests the service construction
        assert_eq!(request.method, HttpMethod::Get);
        assert!(request.url.contains("test"));
    }

    #[test]
    fn test_ml_api_patterns() {
        let service = ApiService::with_mock();

        let features = json!({"feature1": 1.0, "feature2": 2.0});
        let request_builder =
            MLApiPatterns::prediction_request(&service, "predict", &features).unwrap();
        let request = request_builder.build();

        assert_eq!(request.method, HttpMethod::Post);
        assert!(request.url.contains("predict"));
        assert!(request.body.is_some());
    }

    #[test]
    fn test_authentication() {
        let auth = Authentication::Bearer("test-token".to_string());
        let config =
            ApiConfig::new("https://api.example.com".to_string()).with_authentication(auth);

        match config.authentication {
            Some(Authentication::Bearer(token)) => assert_eq!(token, "test-token"),
            _ => panic!("Expected Bearer authentication"),
        }
    }

    #[test]
    fn test_api_metrics() {
        let metrics = ApiMetrics {
            total_requests: 10,
            successful_requests: 8,
            failed_requests: 2,
            ..ApiMetrics::default()
        };

        assert_eq!(metrics.success_rate(), 0.8);

        let display = metrics.to_string();
        assert!(display.contains("Total Requests: 10"));
        assert!(display.contains("Success Rate: 80.00%"));
    }

    #[test]
    fn test_query_param_building() {
        let request = ApiRequest::new(HttpMethod::Get, "https://api.example.com".to_string())
            .with_query_param("param1".to_string(), "value1".to_string())
            .with_query_param("param2".to_string(), "value2".to_string());

        let url = request.build_url();
        assert!(url.contains("param1=value1"));
        assert!(url.contains("param2=value2"));
        assert!(url.contains("?"));
        assert!(url.contains("&"));
    }
}
