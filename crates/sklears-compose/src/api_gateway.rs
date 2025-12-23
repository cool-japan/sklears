use std::sync::Arc;
use std::time::{Duration, Instant};
use scirs2_core::random::thread_rng;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::timeout;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub name: String,
    pub url: String,
    pub health_check_path: String,
    pub timeout: Duration,
    pub retry_count: u32,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteConfig {
    pub path: String,
    pub method: String,
    pub service_name: String,
    pub rate_limit: Option<u32>,
    pub cache_ttl: Option<Duration>,
    pub auth_required: bool,
}

#[derive(Debug, Clone)]
pub struct ServiceHealth {
    pub is_healthy: bool,
    pub last_check: Instant,
    pub response_time: Duration,
    pub error_count: u32,
}

#[derive(Debug)]
pub struct MLPipelineGateway {
    services: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    routes: Arc<RwLock<HashMap<String, RouteConfig>>>,
    health_status: Arc<RwLock<HashMap<String, ServiceHealth>>>,
    rate_limiters: Arc<RwLock<HashMap<String, Arc<Semaphore>>>>,
    cache: Arc<RwLock<HashMap<String, (String, Instant)>>>,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
}

#[derive(Debug, Clone)]
struct CircuitBreaker {
    failure_threshold: u32,
    recovery_timeout: Duration,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    state: CircuitBreakerState,
}

#[derive(Debug, Clone, PartialEq)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("Service not found: {service}")]
    ServiceNotFound { service: String },
    
    #[error("Route not configured: {path}")]
    RouteNotConfigured { path: String },
    
    #[error("Service unhealthy: {service}")]
    ServiceUnhealthy { service: String },
    
    #[error("Rate limit exceeded for service: {service}")]
    RateLimitExceeded { service: String },
    
    #[error("Circuit breaker open for service: {service}")]
    CircuitBreakerOpen { service: String },
    
    #[error("Request timeout for service: {service}")]
    RequestTimeout { service: String },
    
    #[error("Authentication required")]
    AuthenticationRequired,
    
    #[error("Service communication error: {error}")]
    ServiceError { error: String },
}

impl MLPipelineGateway {
    pub fn new() -> Self {
        Self {
            services: Arc::new(RwLock::new(HashMap::new())),
            routes: Arc::new(RwLock::new(HashMap::new())),
            health_status: Arc::new(RwLock::new(HashMap::new())),
            rate_limiters: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_service(&self, service: ServiceEndpoint) -> Result<(), GatewayError> {
        let mut services = self.services.write().await;
        let mut rate_limiters = self.rate_limiters.write().await;
        let mut circuit_breakers = self.circuit_breakers.write().await;
        
        services.insert(service.name.clone(), service.clone());
        
        if let Some(limit) = self.get_default_rate_limit(&service.name) {
            rate_limiters.insert(
                service.name.clone(),
                Arc::new(Semaphore::new(limit as usize))
            );
        }
        
        circuit_breakers.insert(
            service.name.clone(),
            CircuitBreaker {
                failure_threshold: 5,
                recovery_timeout: Duration::from_secs(30),
                failure_count: 0,
                last_failure_time: None,
                state: CircuitBreakerState::Closed,
            }
        );
        
        Ok(())
    }

    pub async fn register_route(&self, route: RouteConfig) -> Result<(), GatewayError> {
        let mut routes = self.routes.write().await;
        let route_key = format!("{}:{}", route.method, route.path);
        routes.insert(route_key, route);
        Ok(())
    }

    pub async fn route_request(
        &self,
        method: &str,
        path: &str,
        body: &str,
        headers: HashMap<String, String>,
    ) -> Result<String, GatewayError> {
        let route_key = format!("{}:{}", method, path);
        let routes = self.routes.read().await;
        
        let route = routes.get(&route_key)
            .ok_or(GatewayError::RouteNotConfigured { path: path.to_string() })?;
        
        if route.auth_required && !self.is_authenticated(&headers) {
            return Err(GatewayError::AuthenticationRequired);
        }
        
        let cache_key = format!("{}:{}:{}", method, path, body);
        if let Some(cached_response) = self.get_cached_response(&cache_key, route.cache_ttl).await {
            return Ok(cached_response);
        }
        
        self.check_rate_limit(&route.service_name).await?;
        self.check_circuit_breaker(&route.service_name).await?;
        self.check_service_health(&route.service_name).await?;
        
        let response = self.forward_request(&route.service_name, method, path, body, headers).await?;
        
        if let Some(ttl) = route.cache_ttl {
            self.cache_response(&cache_key, &response, ttl).await;
        }
        
        Ok(response)
    }

    async fn check_rate_limit(&self, service_name: &str) -> Result<(), GatewayError> {
        let rate_limiters = self.rate_limiters.read().await;
        if let Some(semaphore) = rate_limiters.get(service_name) {
            match semaphore.clone().try_acquire() {
                Ok(_permit) => Ok(()),
                Err(_) => Err(GatewayError::RateLimitExceeded { 
                    service: service_name.to_string() 
                }),
            }
        } else {
            Ok(())
        }
    }

    async fn check_circuit_breaker(&self, service_name: &str) -> Result<(), GatewayError> {
        let mut circuit_breakers = self.circuit_breakers.write().await;
        if let Some(cb) = circuit_breakers.get_mut(service_name) {
            match cb.state {
                CircuitBreakerState::Open => {
                    if let Some(last_failure) = cb.last_failure_time {
                        if last_failure.elapsed() > cb.recovery_timeout {
                            cb.state = CircuitBreakerState::HalfOpen;
                            Ok(())
                        } else {
                            Err(GatewayError::CircuitBreakerOpen { 
                                service: service_name.to_string() 
                            })
                        }
                    } else {
                        cb.state = CircuitBreakerState::HalfOpen;
                        Ok(())
                    }
                },
                CircuitBreakerState::HalfOpen | CircuitBreakerState::Closed => Ok(()),
            }
        } else {
            Ok(())
        }
    }

    async fn check_service_health(&self, service_name: &str) -> Result<(), GatewayError> {
        let health_status = self.health_status.read().await;
        if let Some(health) = health_status.get(service_name) {
            if health.is_healthy {
                Ok(())
            } else {
                Err(GatewayError::ServiceUnhealthy { 
                    service: service_name.to_string() 
                })
            }
        } else {
            self.perform_health_check(service_name).await
        }
    }

    async fn forward_request(
        &self,
        service_name: &str,
        method: &str,
        path: &str,
        body: &str,
        headers: HashMap<String, String>,
    ) -> Result<String, GatewayError> {
        let services = self.services.read().await;
        let service = services.get(service_name)
            .ok_or(GatewayError::ServiceNotFound { 
                service: service_name.to_string() 
            })?;

        let request_timeout = service.timeout;
        let url = format!("{}{}", service.url, path);
        
        let start_time = Instant::now();
        
        match timeout(request_timeout, self.make_http_request(&url, method, body, headers)).await {
            Ok(Ok(response)) => {
                self.record_success(service_name, start_time.elapsed()).await;
                Ok(response)
            },
            Ok(Err(error)) => {
                self.record_failure(service_name).await;
                Err(GatewayError::ServiceError { error })
            },
            Err(_) => {
                self.record_failure(service_name).await;
                Err(GatewayError::RequestTimeout { 
                    service: service_name.to_string() 
                })
            }
        }
    }

    async fn make_http_request(
        &self,
        url: &str,
        method: &str,
        body: &str,
        headers: HashMap<String, String>,
    ) -> Result<String, String> {
        let client = reqwest::Client::new();
        let mut request_builder = match method.to_uppercase().as_str() {
            "GET" => client.get(url),
            "POST" => client.post(url).body(body.to_string()),
            "PUT" => client.put(url).body(body.to_string()),
            "DELETE" => client.delete(url),
            _ => return Err(format!("Unsupported HTTP method: {}", method)),
        };

        for (key, value) in headers {
            request_builder = request_builder.header(&key, &value);
        }

        match request_builder.send().await {
            Ok(response) => {
                if response.status().is_success() {
                    match response.text().await {
                        Ok(text) => Ok(text),
                        Err(e) => Err(format!("Failed to read response body: {}", e)),
                    }
                } else {
                    Err(format!("HTTP error: {}", response.status()))
                }
            },
            Err(e) => Err(format!("Request failed: {}", e)),
        }
    }

    async fn perform_health_check(&self, service_name: &str) -> Result<(), GatewayError> {
        let services = self.services.read().await;
        let service = services.get(service_name)
            .ok_or(GatewayError::ServiceNotFound { 
                service: service_name.to_string() 
            })?;

        let health_url = format!("{}{}", service.url, service.health_check_path);
        let start_time = Instant::now();
        
        drop(services);

        match timeout(Duration::from_secs(5), self.make_http_request(&health_url, "GET", "", HashMap::new())).await {
            Ok(Ok(_)) => {
                let response_time = start_time.elapsed();
                let mut health_status = self.health_status.write().await;
                health_status.insert(service_name.to_string(), ServiceHealth {
                    is_healthy: true,
                    last_check: Instant::now(),
                    response_time,
                    error_count: 0,
                });
                Ok(())
            },
            Ok(Err(_)) | Err(_) => {
                let mut health_status = self.health_status.write().await;
                health_status.insert(service_name.to_string(), ServiceHealth {
                    is_healthy: false,
                    last_check: Instant::now(),
                    response_time: start_time.elapsed(),
                    error_count: 1,
                });
                Err(GatewayError::ServiceUnhealthy { 
                    service: service_name.to_string() 
                })
            }
        }
    }

    async fn record_success(&self, service_name: &str, response_time: Duration) {
        let mut circuit_breakers = self.circuit_breakers.write().await;
        if let Some(cb) = circuit_breakers.get_mut(service_name) {
            cb.failure_count = 0;
            if cb.state == CircuitBreakerState::HalfOpen {
                cb.state = CircuitBreakerState::Closed;
            }
        }

        let mut health_status = self.health_status.write().await;
        if let Some(health) = health_status.get_mut(service_name) {
            health.is_healthy = true;
            health.response_time = response_time;
            health.error_count = 0;
        }
    }

    async fn record_failure(&self, service_name: &str) {
        let mut circuit_breakers = self.circuit_breakers.write().await;
        if let Some(cb) = circuit_breakers.get_mut(service_name) {
            cb.failure_count += 1;
            cb.last_failure_time = Some(Instant::now());
            
            if cb.failure_count >= cb.failure_threshold {
                cb.state = CircuitBreakerState::Open;
            }
        }

        let mut health_status = self.health_status.write().await;
        if let Some(health) = health_status.get_mut(service_name) {
            health.is_healthy = false;
            health.error_count += 1;
        }
    }

    async fn get_cached_response(&self, cache_key: &str, ttl: Option<Duration>) -> Option<String> {
        if let Some(ttl) = ttl {
            let cache = self.cache.read().await;
            if let Some((response, cached_at)) = cache.get(cache_key) {
                if cached_at.elapsed() < ttl {
                    return Some(response.clone());
                }
            }
        }
        None
    }

    async fn cache_response(&self, cache_key: &str, response: &str, ttl: Duration) {
        let mut cache = self.cache.write().await;
        cache.insert(cache_key.to_string(), (response.to_string(), Instant::now()));
        
        tokio::spawn({
            let cache = self.cache.clone();
            let key = cache_key.to_string();
            async move {
                tokio::time::sleep(ttl).await;
                let mut cache = cache.write().await;
                cache.remove(&key);
            }
        });
    }

    fn is_authenticated(&self, headers: &HashMap<String, String>) -> bool {
        headers.contains_key("authorization") || headers.contains_key("Authorization")
    }

    fn get_default_rate_limit(&self, _service_name: &str) -> Option<u32> {
        Some(100)
    }

    pub async fn get_service_metrics(&self) -> HashMap<String, ServiceMetrics> {
        let mut metrics = HashMap::new();
        let health_status = self.health_status.read().await;
        let circuit_breakers = self.circuit_breakers.read().await;
        
        for (service_name, health) in health_status.iter() {
            let cb_state = circuit_breakers.get(service_name)
                .map(|cb| cb.state.clone())
                .unwrap_or(CircuitBreakerState::Closed);
                
            metrics.insert(service_name.clone(), ServiceMetrics {
                is_healthy: health.is_healthy,
                response_time: health.response_time,
                error_count: health.error_count,
                circuit_breaker_state: cb_state,
                last_health_check: health.last_check,
            });
        }
        
        metrics
    }

    pub async fn start_health_monitor(&self) {
        let gateway = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                gateway.run_health_checks().await;
            }
        });
    }

    async fn run_health_checks(&self) {
        let services: Vec<String> = {
            let services = self.services.read().await;
            services.keys().cloned().collect()
        };
        
        for service_name in services {
            if let Err(e) = self.perform_health_check(&service_name).await {
                log::warn!("Health check failed for service {}: {:?}", service_name, e);
            }
        }
    }
}

impl Clone for MLPipelineGateway {
    fn clone(&self) -> Self {
        Self {
            services: self.services.clone(),
            routes: self.routes.clone(),
            health_status: self.health_status.clone(),
            rate_limiters: self.rate_limiters.clone(),
            cache: self.cache.clone(),
            circuit_breakers: self.circuit_breakers.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServiceMetrics {
    pub is_healthy: bool,
    pub response_time: Duration,
    pub error_count: u32,
    pub circuit_breaker_state: CircuitBreakerState,
    pub last_health_check: Instant,
}

pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin { current: std::sync::Arc<std::sync::atomic::AtomicUsize> },
    WeightedRoundRobin,
    LeastConnections,
    Random,
    HealthBased,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self { strategy }
    }
    
    pub async fn select_service(
        &self,
        services: &[ServiceEndpoint],
        health_status: &HashMap<String, ServiceHealth>,
    ) -> Option<&ServiceEndpoint> {
        let healthy_services: Vec<_> = services.iter()
            .filter(|service| {
                health_status.get(&service.name)
                    .map(|health| health.is_healthy)
                    .unwrap_or(false)
            })
            .collect();
            
        if healthy_services.is_empty() {
            return None;
        }
        
        match &self.strategy {
            LoadBalancingStrategy::RoundRobin { current } => {
                let index = current.fetch_add(1, std::sync::atomic::Ordering::Relaxed) 
                    % healthy_services.len();
                Some(healthy_services[index])
            },
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.weighted_selection(&healthy_services)
            },
            LoadBalancingStrategy::Random => {
                let mut rng = thread_rng();
                let index = rng.gen_range(0..healthy_services.len());
                Some(healthy_services[index])
            },
            LoadBalancingStrategy::LeastConnections => {
                healthy_services.iter()
                    .min_by_key(|service| {
                        health_status.get(&service.name)
                            .map(|health| health.error_count)
                            .unwrap_or(u32::MAX)
                    })
                    .copied()
            },
            LoadBalancingStrategy::HealthBased => {
                healthy_services.iter()
                    .min_by_key(|service| {
                        health_status.get(&service.name)
                            .map(|health| health.response_time.as_millis() as u64)
                            .unwrap_or(u64::MAX)
                    })
                    .copied()
            },
        }
    }
    
    fn weighted_selection(&self, services: &[&ServiceEndpoint]) -> Option<&ServiceEndpoint> {
        let total_weight: f64 = services.iter().map(|s| s.weight).sum();
        if total_weight <= 0.0 {
            return services.first().copied();
        }
        
        let mut rng = thread_rng();
        let mut random_weight = rng.gen() * total_weight;
        
        for service in services {
            random_weight -= service.weight;
            if random_weight <= 0.0 {
                return Some(service);
            }
        }
        
        services.last().copied()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_service_registration() {
        let gateway = MLPipelineGateway::new();
        
        let service = ServiceEndpoint {
            name: "ml-model-service".to_string(),
            url: "http://localhost:8080".to_string(),
            health_check_path: "/health".to_string(),
            timeout: Duration::from_secs(30),
            retry_count: 3,
            weight: 1.0,
        };
        
        assert!(gateway.register_service(service).await.is_ok());
    }

    #[test]
    async fn test_route_registration() {
        let gateway = MLPipelineGateway::new();
        
        let route = RouteConfig {
            path: "/predict".to_string(),
            method: "POST".to_string(),
            service_name: "ml-model-service".to_string(),
            rate_limit: Some(100),
            cache_ttl: Some(Duration::from_secs(300)),
            auth_required: true,
        };
        
        assert!(gateway.register_route(route).await.is_ok());
    }

    #[test]
    async fn test_load_balancer_round_robin() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin {
            current: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0))
        });
        
        let services = vec![
            ServiceEndpoint {
                name: "service1".to_string(),
                url: "http://localhost:8081".to_string(),
                health_check_path: "/health".to_string(),
                timeout: Duration::from_secs(30),
                retry_count: 3,
                weight: 1.0,
            },
            ServiceEndpoint {
                name: "service2".to_string(),
                url: "http://localhost:8082".to_string(),
                health_check_path: "/health".to_string(),
                timeout: Duration::from_secs(30),
                retry_count: 3,
                weight: 1.0,
            },
        ];
        
        let mut health_status = HashMap::new();
        health_status.insert("service1".to_string(), ServiceHealth {
            is_healthy: true,
            last_check: Instant::now(),
            response_time: Duration::from_millis(100),
            error_count: 0,
        });
        health_status.insert("service2".to_string(), ServiceHealth {
            is_healthy: true,
            last_check: Instant::now(),
            response_time: Duration::from_millis(150),
            error_count: 0,
        });
        
        let selected1 = lb.select_service(&services, &health_status).await;
        let selected2 = lb.select_service(&services, &health_status).await;
        
        assert!(selected1.is_some());
        assert!(selected2.is_some());
        assert_ne!(selected1.unwrap().name, selected2.unwrap().name);
    }

    #[test]
    async fn test_circuit_breaker_functionality() {
        let gateway = MLPipelineGateway::new();
        
        gateway.record_failure("test-service").await;
        gateway.record_failure("test-service").await;
        gateway.record_failure("test-service").await;
        gateway.record_failure("test-service").await;
        gateway.record_failure("test-service").await;
        
        let result = gateway.check_circuit_breaker("test-service").await;
        assert!(matches!(result, Err(GatewayError::CircuitBreakerOpen { .. })));
    }

    #[test]
    async fn test_service_metrics() {
        let gateway = MLPipelineGateway::new();
        
        let service = ServiceEndpoint {
            name: "test-service".to_string(),
            url: "http://localhost:8080".to_string(),
            health_check_path: "/health".to_string(),
            timeout: Duration::from_secs(30),
            retry_count: 3,
            weight: 1.0,
        };
        
        gateway.register_service(service).await.unwrap();
        
        let mut health_status = gateway.health_status.write().await;
        health_status.insert("test-service".to_string(), ServiceHealth {
            is_healthy: true,
            last_check: Instant::now(),
            response_time: Duration::from_millis(200),
            error_count: 0,
        });
        drop(health_status);
        
        let metrics = gateway.get_service_metrics().await;
        assert!(metrics.contains_key("test-service"));
        assert!(metrics["test-service"].is_healthy);
    }
}