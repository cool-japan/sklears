//! Database connectivity utilities for ML data processing
//!
//! This module provides comprehensive database connection and query utilities for machine learning
//! workflows, including connection pooling, query building, and data conversion.

use crate::UtilsError;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub pool_size: usize,
    pub connection_timeout: Duration,
    pub query_timeout: Duration,
    pub ssl_mode: SslMode,
    pub additional_params: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SslMode {
    Disable,
    Prefer,
    Require,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5432,
            database: "postgres".to_string(),
            username: "postgres".to_string(),
            password: String::new(),
            pool_size: 10,
            connection_timeout: Duration::from_secs(30),
            query_timeout: Duration::from_secs(60),
            ssl_mode: SslMode::Prefer,
            additional_params: HashMap::new(),
        }
    }
}

impl DatabaseConfig {
    pub fn new(host: String, database: String, username: String, password: String) -> Self {
        Self {
            host,
            database,
            username,
            password,
            ..Default::default()
        }
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_pool_size(mut self, pool_size: usize) -> Self {
        self.pool_size = pool_size;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.connection_timeout = timeout;
        self.query_timeout = timeout;
        self
    }

    pub fn connection_string(&self) -> String {
        let ssl_param = match self.ssl_mode {
            SslMode::Disable => "sslmode=disable",
            SslMode::Prefer => "sslmode=prefer",
            SslMode::Require => "sslmode=require",
        };

        let mut params = vec![
            format!("host={}", self.host),
            format!("port={}", self.port),
            format!("dbname={}", self.database),
            format!("user={}", self.username),
            ssl_param.to_string(),
        ];

        if !self.password.is_empty() {
            params.push(format!("password={}", self.password));
        }

        for (key, value) in &self.additional_params {
            params.push(format!("{key}={value}"));
        }

        params.join(" ")
    }
}

/// Database-specific error types
#[derive(thiserror::Error, Debug, Clone)]
pub enum DatabaseError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Query execution failed: {0}")]
    QueryFailed(String),
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),
    #[error("Data conversion failed: {0}")]
    ConversionFailed(String),
    #[error("Connection pool exhausted")]
    PoolExhausted,
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

impl From<DatabaseError> for UtilsError {
    fn from(err: DatabaseError) -> Self {
        UtilsError::InvalidParameter(err.to_string())
    }
}

/// Represents a single database row result
#[derive(Debug, Clone)]
pub struct Row {
    columns: HashMap<String, Value>,
    column_order: Vec<String>,
}

impl Row {
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            column_order: Vec::new(),
        }
    }

    pub fn insert<T: Into<Value>>(&mut self, column: String, value: T) {
        if !self.columns.contains_key(&column) {
            self.column_order.push(column.clone());
        }
        self.columns.insert(column, value.into());
    }

    pub fn get(&self, column: &str) -> Option<&Value> {
        self.columns.get(column)
    }

    pub fn get_string(&self, column: &str) -> Option<String> {
        self.get(column)?.as_string()
    }

    pub fn get_f64(&self, column: &str) -> Option<f64> {
        self.get(column)?.as_f64()
    }

    pub fn get_i64(&self, column: &str) -> Option<i64> {
        self.get(column)?.as_i64()
    }

    pub fn columns(&self) -> &[String] {
        &self.column_order
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}

/// Generic database value type
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
}

impl Value {
    pub fn as_string(&self) -> Option<String> {
        match self {
            Value::String(s) => Some(s.clone()),
            Value::Int(i) => Some(i.to_string()),
            Value::Float(f) => Some(f.to_string()),
            Value::Bool(b) => Some(b.to_string()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Int(i as i64)
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}

impl From<f32> for Value {
    fn from(f: f32) -> Self {
        Value::Float(f as f64)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

/// Database connection trait
pub trait Connection {
    fn execute(&self, query: &Query) -> Result<QueryResult, DatabaseError>;
    fn query(&self, query: &Query) -> Result<ResultSet, DatabaseError>;
    fn begin_transaction(&self) -> Result<Transaction, DatabaseError>;
    fn close(&self) -> Result<(), DatabaseError>;
    fn is_connected(&self) -> bool;
}

/// Mock database connection for testing
pub struct MockConnection {
    connected: bool,
    mock_data: HashMap<String, Vec<Row>>,
}

impl MockConnection {
    pub fn new() -> Self {
        Self {
            connected: true,
            mock_data: HashMap::new(),
        }
    }

    pub fn add_mock_data(&mut self, table: String, rows: Vec<Row>) {
        self.mock_data.insert(table, rows);
    }
}

impl Default for MockConnection {
    fn default() -> Self {
        Self::new()
    }
}

impl Connection for MockConnection {
    fn execute(&self, _query: &Query) -> Result<QueryResult, DatabaseError> {
        if !self.connected {
            return Err(DatabaseError::ConnectionFailed("Not connected".to_string()));
        }

        Ok(QueryResult {
            rows_affected: 1,
            execution_time: Duration::from_millis(10),
        })
    }

    fn query(&self, _query: &Query) -> Result<ResultSet, DatabaseError> {
        if !self.connected {
            return Err(DatabaseError::ConnectionFailed("Not connected".to_string()));
        }

        // Simple mock: return empty result set
        let mut result = ResultSet::new(vec!["id".to_string(), "value".to_string()]);
        result.set_execution_time(Duration::from_millis(5));
        Ok(result)
    }

    fn begin_transaction(&self) -> Result<Transaction, DatabaseError> {
        if !self.connected {
            return Err(DatabaseError::ConnectionFailed("Not connected".to_string()));
        }
        Ok(Transaction::new())
    }

    fn close(&self) -> Result<(), DatabaseError> {
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

/// Database connection pool
pub struct DatabasePool {
    #[allow(dead_code)]
    config: DatabaseConfig,
    connections: Arc<Mutex<Vec<Box<dyn Connection + Send + Sync>>>>,
    max_size: usize,
}

impl DatabasePool {
    pub fn new(config: DatabaseConfig) -> Self {
        let max_size = config.pool_size;
        Self {
            config,
            connections: Arc::new(Mutex::new(Vec::new())),
            max_size,
        }
    }

    pub fn get_connection(&self) -> Result<Box<dyn Connection + Send + Sync>, DatabaseError> {
        // For now, return a mock connection
        // In a real implementation, this would manage actual database connections
        Ok(Box::new(MockConnection::new()))
    }

    pub fn return_connection(&self, _connection: Box<dyn Connection + Send + Sync>) {
        // Return connection to pool
    }

    pub fn size(&self) -> usize {
        self.connections.lock().unwrap().len()
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

/// Query builder for constructing SQL queries
pub struct QueryBuilder {
    query_type: QueryType,
    table: Option<String>,
    columns: Vec<String>,
    conditions: Vec<String>,
    joins: Vec<String>,
    order_by: Vec<String>,
    group_by: Vec<String>,
    having: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    parameters: Vec<Value>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum QueryType {
    Select,
    Insert,
    Update,
    Delete,
}

impl QueryBuilder {
    pub fn select() -> Self {
        Self {
            query_type: QueryType::Select,
            table: None,
            columns: Vec::new(),
            conditions: Vec::new(),
            joins: Vec::new(),
            order_by: Vec::new(),
            group_by: Vec::new(),
            having: Vec::new(),
            limit: None,
            offset: None,
            parameters: Vec::new(),
        }
    }

    pub fn from(mut self, table: &str) -> Self {
        self.table = Some(table.to_string());
        self
    }

    pub fn columns(mut self, columns: &[&str]) -> Self {
        self.columns = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn where_clause(mut self, condition: &str) -> Self {
        self.conditions.push(condition.to_string());
        self
    }

    pub fn join(mut self, join_clause: &str) -> Self {
        self.joins.push(join_clause.to_string());
        self
    }

    pub fn order_by(mut self, column: &str, ascending: bool) -> Self {
        let direction = if ascending { "ASC" } else { "DESC" };
        self.order_by.push(format!("{column} {direction}"));
        self
    }

    pub fn group_by(mut self, columns: &[&str]) -> Self {
        self.group_by = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn parameter<T: Into<Value>>(mut self, value: T) -> Self {
        self.parameters.push(value.into());
        self
    }

    pub fn build(self) -> Query {
        let sql = self.build_sql();
        Query::new(sql, self.parameters)
    }

    fn build_sql(&self) -> String {
        match self.query_type {
            QueryType::Select => self.build_select(),
            _ => "".to_string(), // Other query types can be implemented as needed
        }
    }

    fn build_select(&self) -> String {
        let mut query = String::new();

        // SELECT clause
        query.push_str("SELECT ");
        if self.columns.is_empty() {
            query.push('*');
        } else {
            query.push_str(&self.columns.join(", "));
        }

        // FROM clause
        if let Some(table) = &self.table {
            query.push_str(&format!(" FROM {table}"));
        }

        // JOINs
        for join in &self.joins {
            query.push_str(&format!(" {join}"));
        }

        // WHERE clause
        if !self.conditions.is_empty() {
            query.push_str(&format!(" WHERE {}", self.conditions.join(" AND ")));
        }

        // GROUP BY clause
        if !self.group_by.is_empty() {
            query.push_str(&format!(" GROUP BY {}", self.group_by.join(", ")));
        }

        // HAVING clause
        if !self.having.is_empty() {
            query.push_str(&format!(" HAVING {}", self.having.join(" AND ")));
        }

        // ORDER BY clause
        if !self.order_by.is_empty() {
            query.push_str(&format!(" ORDER BY {}", self.order_by.join(", ")));
        }

        // LIMIT clause
        if let Some(limit) = self.limit {
            query.push_str(&format!(" LIMIT {limit}"));
        }

        // OFFSET clause
        if let Some(offset) = self.offset {
            query.push_str(&format!(" OFFSET {offset}"));
        }

        query
    }
}

/// Represents a SQL query with parameters
#[derive(Debug, Clone)]
pub struct Query {
    sql: String,
    parameters: Vec<Value>,
}

impl Query {
    pub fn new(sql: String, parameters: Vec<Value>) -> Self {
        Self { sql, parameters }
    }

    pub fn sql(&self) -> &str {
        &self.sql
    }

    pub fn parameters(&self) -> &[Value] {
        &self.parameters
    }
}

/// Query execution result
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub rows_affected: usize,
    pub execution_time: Duration,
}

/// Database query result set
#[derive(Debug, Clone)]
pub struct ResultSet {
    rows: Vec<Row>,
    columns: Vec<String>,
    execution_time: Duration,
    #[allow(dead_code)]
    rows_affected: Option<usize>,
}

impl ResultSet {
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            rows: Vec::new(),
            columns,
            execution_time: Duration::from_secs(0),
            rows_affected: None,
        }
    }

    pub fn add_row(&mut self, row: Row) {
        self.rows.push(row);
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn execution_time(&self) -> Duration {
        self.execution_time
    }

    pub fn set_execution_time(&mut self, time: Duration) {
        self.execution_time = time;
    }

    /// Convert result set to ndarray `Array2<f64>` for ML processing
    pub fn to_array2(&self) -> Result<Array2<f64>, DatabaseError> {
        if self.rows.is_empty() {
            return Err(DatabaseError::ConversionFailed(
                "Cannot convert empty result set to array".to_string(),
            ));
        }

        let n_rows = self.rows.len();
        let n_cols = self.columns.len();
        let mut data = Array2::zeros((n_rows, n_cols));

        for (row_idx, row) in self.rows.iter().enumerate() {
            for (col_idx, col_name) in self.columns.iter().enumerate() {
                let value = row.get(col_name).ok_or_else(|| {
                    DatabaseError::ConversionFailed(format!("Column '{col_name}' not found in row"))
                })?;

                let numeric_value = value.as_f64().ok_or_else(|| {
                    DatabaseError::ConversionFailed(format!(
                        "Cannot convert value to f64: {value:?}"
                    ))
                })?;

                data[[row_idx, col_idx]] = numeric_value;
            }
        }

        Ok(data)
    }

    /// Convert specific column to `Array1<f64>`
    pub fn column_to_array1(&self, column: &str) -> Result<Array1<f64>, DatabaseError> {
        if !self.columns.contains(&column.to_string()) {
            return Err(DatabaseError::ConversionFailed(format!(
                "Column '{column}' not found"
            )));
        }

        let mut data = Array1::zeros(self.rows.len());
        for (idx, row) in self.rows.iter().enumerate() {
            let value = row.get(column).ok_or_else(|| {
                DatabaseError::ConversionFailed(format!("Column '{column}' not found in row"))
            })?;

            let numeric_value = value.as_f64().ok_or_else(|| {
                DatabaseError::ConversionFailed(format!("Cannot convert value to f64: {value:?}"))
            })?;

            data[idx] = numeric_value;
        }

        Ok(data)
    }

    /// Get unique values from a column
    pub fn unique_values(&self, column: &str) -> Result<Vec<Value>, DatabaseError> {
        if !self.columns.contains(&column.to_string()) {
            return Err(DatabaseError::ConversionFailed(format!(
                "Column '{column}' not found"
            )));
        }

        let mut unique_values = Vec::new();
        for row in &self.rows {
            if let Some(value) = row.get(column) {
                if !unique_values.contains(value) {
                    unique_values.push(value.clone());
                }
            }
        }

        Ok(unique_values)
    }
}

/// Database transaction
pub struct Transaction {
    committed: bool,
    rolled_back: bool,
}

impl Transaction {
    pub fn new() -> Self {
        Self {
            committed: false,
            rolled_back: false,
        }
    }

    pub fn commit(&mut self) -> Result<(), DatabaseError> {
        if self.rolled_back {
            return Err(DatabaseError::TransactionFailed(
                "Transaction already rolled back".to_string(),
            ));
        }
        self.committed = true;
        Ok(())
    }

    pub fn rollback(&mut self) -> Result<(), DatabaseError> {
        if self.committed {
            return Err(DatabaseError::TransactionFailed(
                "Transaction already committed".to_string(),
            ));
        }
        self.rolled_back = true;
        Ok(())
    }

    pub fn is_committed(&self) -> bool {
        self.committed
    }

    pub fn is_rolled_back(&self) -> bool {
        self.rolled_back
    }
}

impl Default for Transaction {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DatabaseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}@{}:{}/{}",
            self.username, self.host, self.port, self.database
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_config() {
        let config = DatabaseConfig::new(
            "localhost".to_string(),
            "testdb".to_string(),
            "user".to_string(),
            "pass".to_string(),
        )
        .with_port(3306)
        .with_pool_size(5);

        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 3306);
        assert_eq!(config.pool_size, 5);

        let conn_str = config.connection_string();
        assert!(conn_str.contains("host=localhost"));
        assert!(conn_str.contains("port=3306"));
    }

    #[test]
    fn test_value_conversions() {
        let int_val = Value::from(42i64);
        assert_eq!(int_val.as_i64(), Some(42));
        assert_eq!(int_val.as_f64(), Some(42.0));

        let float_val = Value::from(std::f64::consts::PI);
        assert_eq!(float_val.as_f64(), Some(std::f64::consts::PI));

        let string_val = Value::from("hello");
        assert_eq!(string_val.as_string(), Some("hello".to_string()));
    }

    #[test]
    fn test_row_operations() {
        let mut row = Row::new();
        row.insert("id".to_string(), 1i64);
        row.insert("name".to_string(), "test");
        row.insert("score".to_string(), 95.5f64);

        assert_eq!(row.get_i64("id"), Some(1));
        assert_eq!(row.get_string("name"), Some("test".to_string()));
        assert_eq!(row.get_f64("score"), Some(95.5));
        assert_eq!(row.columns().len(), 3);
    }

    #[test]
    fn test_result_set_array_conversion() {
        let mut result_set = ResultSet::new(vec!["a".to_string(), "b".to_string()]);

        let mut row1 = Row::new();
        row1.insert("a".to_string(), 1.0f64);
        row1.insert("b".to_string(), 2.0f64);
        result_set.add_row(row1);

        let mut row2 = Row::new();
        row2.insert("a".to_string(), 3.0f64);
        row2.insert("b".to_string(), 4.0f64);
        result_set.add_row(row2);

        let array = result_set.to_array2().unwrap();
        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array[[0, 0]], 1.0);
        assert_eq!(array[[1, 1]], 4.0);
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::select()
            .columns(&["id", "name", "score"])
            .from("users")
            .where_clause("score > 80")
            .order_by("score", false)
            .limit(10)
            .build();

        let sql = query.sql();
        assert!(sql.contains("SELECT id, name, score"));
        assert!(sql.contains("FROM users"));
        assert!(sql.contains("WHERE score > 80"));
        assert!(sql.contains("ORDER BY score DESC"));
        assert!(sql.contains("LIMIT 10"));
    }

    #[test]
    fn test_mock_connection() {
        let connection = MockConnection::new();
        assert!(connection.is_connected());

        let query = Query::new("SELECT 1".to_string(), vec![]);
        let result = connection.execute(&query).unwrap();
        assert_eq!(result.rows_affected, 1);

        let result_set = connection.query(&query).unwrap();
        assert_eq!(result_set.columns().len(), 2);
    }

    #[test]
    fn test_transaction() {
        let mut transaction = Transaction::new();
        assert!(!transaction.is_committed());
        assert!(!transaction.is_rolled_back());

        transaction.commit().unwrap();
        assert!(transaction.is_committed());

        // Cannot rollback after commit
        assert!(transaction.rollback().is_err());
    }

    #[test]
    fn test_database_pool() {
        let config = DatabaseConfig::default();
        let pool = DatabasePool::new(config);

        assert_eq!(pool.max_size(), 10);

        let connection = pool.get_connection().unwrap();
        assert!(connection.is_connected());
    }

    #[test]
    fn test_result_set_unique_values() {
        let mut result_set = ResultSet::new(vec!["category".to_string()]);

        let mut row1 = Row::new();
        row1.insert("category".to_string(), "A");
        result_set.add_row(row1);

        let mut row2 = Row::new();
        row2.insert("category".to_string(), "B");
        result_set.add_row(row2);

        let mut row3 = Row::new();
        row3.insert("category".to_string(), "A");
        result_set.add_row(row3);

        let unique_values = result_set.unique_values("category").unwrap();
        assert_eq!(unique_values.len(), 2);
        assert!(unique_values.contains(&Value::String("A".to_string())));
        assert!(unique_values.contains(&Value::String("B".to_string())));
    }
}
