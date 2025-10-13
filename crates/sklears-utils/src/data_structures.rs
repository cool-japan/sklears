//! Data structures for efficient machine learning operations
//!
//! This module provides specialized data structures optimized for ML workloads,
//! including graph structures, tree structures, and memory-efficient collections.

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Zero;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::Hash;
use std::sync::{Mutex, RwLock};

// ===== GRAPH STRUCTURES =====

/// Graph representation using adjacency lists
#[derive(Clone, Debug)]
pub struct Graph<T> {
    vertices: Vec<T>,
    adjacency: Vec<Vec<usize>>,
    vertex_map: HashMap<T, usize>,
}

impl<T: Clone + Eq + Hash> Default for Graph<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Eq + Hash> Graph<T> {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            adjacency: Vec::new(),
            vertex_map: HashMap::new(),
        }
    }

    /// Add a vertex to the graph
    pub fn add_vertex(&mut self, vertex: T) -> usize {
        if let Some(&idx) = self.vertex_map.get(&vertex) {
            return idx;
        }

        let idx = self.vertices.len();
        self.vertices.push(vertex.clone());
        self.adjacency.push(Vec::new());
        self.vertex_map.insert(vertex, idx);
        idx
    }

    /// Add an edge between two vertices
    pub fn add_edge(&mut self, from: &T, to: &T) -> UtilsResult<()> {
        let from_idx = self
            .vertex_map
            .get(from)
            .ok_or_else(|| UtilsError::InvalidParameter("From vertex not found".to_string()))?;
        let to_idx = self
            .vertex_map
            .get(to)
            .ok_or_else(|| UtilsError::InvalidParameter("To vertex not found".to_string()))?;

        self.adjacency[*from_idx].push(*to_idx);
        Ok(())
    }

    /// Add an undirected edge between two vertices
    pub fn add_undirected_edge(&mut self, v1: &T, v2: &T) -> UtilsResult<()> {
        self.add_edge(v1, v2)?;
        self.add_edge(v2, v1)?;
        Ok(())
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, vertex: &T) -> UtilsResult<Vec<&T>> {
        let idx = self
            .vertex_map
            .get(vertex)
            .ok_or_else(|| UtilsError::InvalidParameter("Vertex not found".to_string()))?;

        Ok(self.adjacency[*idx]
            .iter()
            .map(|&i| &self.vertices[i])
            .collect())
    }

    /// Get number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).sum()
    }

    /// Breadth-first search from a starting vertex
    pub fn bfs(&self, start: &T) -> UtilsResult<Vec<&T>> {
        let start_idx = self
            .vertex_map
            .get(start)
            .ok_or_else(|| UtilsError::InvalidParameter("Start vertex not found".to_string()))?;

        let mut visited = vec![false; self.vertices.len()];
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        queue.push_back(*start_idx);
        visited[*start_idx] = true;

        while let Some(current) = queue.pop_front() {
            result.push(&self.vertices[current]);

            for &neighbor in &self.adjacency[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(result)
    }

    /// Depth-first search from a starting vertex
    pub fn dfs(&self, start: &T) -> UtilsResult<Vec<&T>> {
        let start_idx = self
            .vertex_map
            .get(start)
            .ok_or_else(|| UtilsError::InvalidParameter("Start vertex not found".to_string()))?;

        let mut visited = vec![false; self.vertices.len()];
        let mut result = Vec::new();

        self.dfs_recursive(*start_idx, &mut visited, &mut result);
        Ok(result)
    }

    fn dfs_recursive<'a>(
        &'a self,
        current: usize,
        visited: &mut Vec<bool>,
        result: &mut Vec<&'a T>,
    ) {
        visited[current] = true;
        result.push(&self.vertices[current]);

        for &neighbor in &self.adjacency[current] {
            if !visited[neighbor] {
                self.dfs_recursive(neighbor, visited, result);
            }
        }
    }

    /// Check if the graph has a cycle (for directed graphs)
    pub fn has_cycle(&self) -> bool {
        let mut visited = vec![false; self.vertices.len()];
        let mut rec_stack = vec![false; self.vertices.len()];

        for i in 0..self.vertices.len() {
            if !visited[i] && self.has_cycle_util(i, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        false
    }

    fn has_cycle_util(
        &self,
        current: usize,
        visited: &mut Vec<bool>,
        rec_stack: &mut Vec<bool>,
    ) -> bool {
        visited[current] = true;
        rec_stack[current] = true;

        for &neighbor in &self.adjacency[current] {
            if (!visited[neighbor] && self.has_cycle_util(neighbor, visited, rec_stack))
                || rec_stack[neighbor]
            {
                return true;
            }
        }

        rec_stack[current] = false;
        false
    }

    /// Convert to adjacency matrix
    pub fn to_adjacency_matrix(&self) -> Array2<u8> {
        let n = self.vertices.len();
        let mut matrix = Array2::zeros((n, n));

        for (i, neighbors) in self.adjacency.iter().enumerate() {
            for &j in neighbors {
                matrix[[i, j]] = 1;
            }
        }

        matrix
    }

    /// Get vertices as a reference
    pub fn vertices(&self) -> &[T] {
        &self.vertices
    }

    /// Get adjacency list as a reference
    pub fn adjacency(&self) -> &[Vec<usize>] {
        &self.adjacency
    }

    /// Serialize the graph to a string representation
    pub fn serialize(&self) -> String
    where
        T: fmt::Display,
    {
        let mut result = String::new();
        result.push_str("Graph {\n");
        result.push_str(&format!("  vertices: {} nodes\n", self.vertices.len()));
        result.push_str(&format!("  edges: {} connections\n", self.num_edges()));
        result.push_str("  adjacency_list: {\n");

        for (i, vertex) in self.vertices.iter().enumerate() {
            result.push_str(&format!("    {vertex}: ["));
            let neighbors: Vec<String> = self.adjacency[i]
                .iter()
                .map(|&idx| self.vertices[idx].to_string())
                .collect();
            result.push_str(&neighbors.join(", "));
            result.push_str("]\n");
        }

        result.push_str("  }\n");
        result.push_str("}\n");
        result
    }

    /// Visualize the graph structure
    pub fn visualize(&self) -> String
    where
        T: fmt::Display,
    {
        let mut result = String::from("Graph Visualization:\n");

        if self.vertices.is_empty() {
            result.push_str("  (empty graph)\n");
            return result;
        }

        result.push_str(&format!("  Vertices: {}\n", self.vertices.len()));
        result.push_str(&format!("  Edges: {}\n", self.num_edges()));
        result.push_str(&format!("  Has cycle: {}\n", self.has_cycle()));
        result.push_str("  Connections:\n");

        for vertex in &self.vertices {
            if let Ok(neighbors) = self.neighbors(vertex) {
                result.push_str(&format!("    {vertex} -> ["));
                let neighbor_strs: Vec<String> = neighbors.iter().map(|n| n.to_string()).collect();
                result.push_str(&neighbor_strs.join(", "));
                result.push_str("]\n");
            }
        }

        result
    }

    /// Compare two graphs for structural equality
    pub fn structural_equals(&self, other: &Self) -> bool
    where
        T: PartialEq,
    {
        if self.vertices.len() != other.vertices.len() {
            return false;
        }

        if self.adjacency.len() != other.adjacency.len() {
            return false;
        }

        // Check if vertices match (order independent)
        for vertex in &self.vertices {
            if !other.vertices.contains(vertex) {
                return false;
            }
        }

        // Check if adjacency relationships match
        for (vertex, neighbors) in self.vertices.iter().zip(self.adjacency.iter()) {
            if let Some(&other_idx) = other.vertex_map.get(vertex) {
                let other_neighbors = &other.adjacency[other_idx];
                if neighbors.len() != other_neighbors.len() {
                    return false;
                }

                // Convert indices to actual vertices for comparison
                let self_neighbor_vertices: HashSet<_> =
                    neighbors.iter().map(|&idx| &self.vertices[idx]).collect();
                let other_neighbor_vertices: HashSet<_> = other_neighbors
                    .iter()
                    .map(|&idx| &other.vertices[idx])
                    .collect();

                if self_neighbor_vertices != other_neighbor_vertices {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

/// Weighted graph representation
#[derive(Clone, Debug)]
pub struct WeightedGraph<T, W> {
    vertices: Vec<T>,
    edges: Vec<Vec<(usize, W)>>,
    vertex_map: HashMap<T, usize>,
}

impl<T: Clone + Eq + Hash, W: Clone + PartialOrd> Default for WeightedGraph<T, W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Eq + Hash, W: Clone + PartialOrd> WeightedGraph<T, W> {
    /// Create a new empty weighted graph
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            vertex_map: HashMap::new(),
        }
    }

    /// Add a vertex to the graph
    pub fn add_vertex(&mut self, vertex: T) -> usize {
        if let Some(&idx) = self.vertex_map.get(&vertex) {
            return idx;
        }

        let idx = self.vertices.len();
        self.vertices.push(vertex.clone());
        self.edges.push(Vec::new());
        self.vertex_map.insert(vertex, idx);
        idx
    }

    /// Add a weighted edge between two vertices
    pub fn add_edge(&mut self, from: &T, to: &T, weight: W) -> UtilsResult<()> {
        let from_idx = self
            .vertex_map
            .get(from)
            .ok_or_else(|| UtilsError::InvalidParameter("From vertex not found".to_string()))?;
        let to_idx = self
            .vertex_map
            .get(to)
            .ok_or_else(|| UtilsError::InvalidParameter("To vertex not found".to_string()))?;

        self.edges[*from_idx].push((*to_idx, weight));
        Ok(())
    }

    /// Get the minimum spanning tree using Kruskal's algorithm
    pub fn minimum_spanning_tree(&self) -> UtilsResult<Vec<(usize, usize, W)>>
    where
        W: Clone + PartialOrd + Copy,
    {
        if self.vertices.is_empty() {
            return Ok(Vec::new());
        }

        let mut edges = Vec::new();
        for (from, adj_list) in self.edges.iter().enumerate() {
            for &(to, weight) in adj_list {
                edges.push((weight, from, to));
            }
        }

        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut parent = (0..self.vertices.len()).collect::<Vec<_>>();
        let mut mst = Vec::new();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);
            if px != py {
                parent[px] = py;
            }
        }

        for (weight, from, to) in edges {
            if find(&mut parent, from) != find(&mut parent, to) {
                union(&mut parent, from, to);
                mst.push((from, to, weight));
            }
        }

        Ok(mst)
    }

    /// Get vertices as a reference
    pub fn vertices(&self) -> &[T] {
        &self.vertices
    }

    /// Get edges as a reference
    pub fn edges(&self) -> &[Vec<(usize, W)>] {
        &self.edges
    }

    /// Serialize the weighted graph to a string representation
    pub fn serialize(&self) -> String
    where
        T: fmt::Display,
        W: fmt::Display,
    {
        let mut result = String::new();
        result.push_str("WeightedGraph {\n");
        result.push_str(&format!("  vertices: {} nodes\n", self.vertices.len()));

        let total_edges: usize = self.edges.iter().map(|edges| edges.len()).sum();
        result.push_str(&format!("  edges: {total_edges} weighted connections\n"));
        result.push_str("  adjacency_list: {\n");

        for (i, vertex) in self.vertices.iter().enumerate() {
            result.push_str(&format!("    {vertex}: ["));
            let edge_strs: Vec<String> = self.edges[i]
                .iter()
                .map(|(to_idx, weight)| format!("{}:{}", self.vertices[*to_idx], weight))
                .collect();
            result.push_str(&edge_strs.join(", "));
            result.push_str("]\n");
        }

        result.push_str("  }\n");
        result.push_str("}\n");
        result
    }

    /// Visualize the weighted graph structure
    pub fn visualize(&self) -> String
    where
        T: fmt::Display,
        W: fmt::Display,
    {
        let mut result = String::from("Weighted Graph Visualization:\n");

        if self.vertices.is_empty() {
            result.push_str("  (empty graph)\n");
            return result;
        }

        let total_edges: usize = self.edges.iter().map(|edges| edges.len()).sum();
        result.push_str(&format!("  Vertices: {}\n", self.vertices.len()));
        result.push_str(&format!("  Weighted Edges: {total_edges}\n"));
        result.push_str("  Connections:\n");

        for (i, vertex) in self.vertices.iter().enumerate() {
            if !self.edges[i].is_empty() {
                result.push_str(&format!("    {vertex} -> ["));
                let edge_strs: Vec<String> = self.edges[i]
                    .iter()
                    .map(|(to_idx, weight)| format!("{}(w:{})", self.vertices[*to_idx], weight))
                    .collect();
                result.push_str(&edge_strs.join(", "));
                result.push_str("]\n");
            } else {
                result.push_str(&format!("    {vertex} -> []\n"));
            }
        }

        result
    }

    /// Compare two weighted graphs for structural equality
    pub fn structural_equals(&self, other: &Self) -> bool
    where
        T: PartialEq,
        W: PartialEq + Eq + Hash,
    {
        if self.vertices.len() != other.vertices.len() {
            return false;
        }

        if self.edges.len() != other.edges.len() {
            return false;
        }

        // Check if vertices match (order independent)
        for vertex in &self.vertices {
            if !other.vertices.contains(vertex) {
                return false;
            }
        }

        // Check if edge relationships match
        for (vertex, edges) in self.vertices.iter().zip(self.edges.iter()) {
            if let Some(&other_idx) = other.vertex_map.get(vertex) {
                let other_edges = &other.edges[other_idx];
                if edges.len() != other_edges.len() {
                    return false;
                }

                // Convert indices to actual vertices with weights for comparison
                let self_edge_set: HashSet<_> = edges
                    .iter()
                    .map(|(idx, weight)| (&self.vertices[*idx], weight))
                    .collect();
                let other_edge_set: HashSet<_> = other_edges
                    .iter()
                    .map(|(idx, weight)| (&other.vertices[*idx], weight))
                    .collect();

                if self_edge_set != other_edge_set {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

// ===== TREE STRUCTURES =====

/// Binary tree node
#[derive(Clone, Debug)]
pub struct TreeNode<T> {
    pub value: T,
    pub left: Option<Box<TreeNode<T>>>,
    pub right: Option<Box<TreeNode<T>>>,
}

impl<T> TreeNode<T> {
    /// Create a new tree node
    pub fn new(value: T) -> Self {
        Self {
            value,
            left: None,
            right: None,
        }
    }

    /// Create a new tree node with children
    pub fn with_children(
        value: T,
        left: Option<Box<TreeNode<T>>>,
        right: Option<Box<TreeNode<T>>>,
    ) -> Self {
        Self { value, left, right }
    }
}

/// Binary Search Tree
#[derive(Clone, Debug)]
pub struct BinarySearchTree<T> {
    root: Option<Box<TreeNode<T>>>,
}

impl<T: Clone + PartialOrd> Default for BinarySearchTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + PartialOrd> BinarySearchTree<T> {
    /// Create a new empty BST
    pub fn new() -> Self {
        Self { root: None }
    }

    /// Insert a value into the BST
    pub fn insert(&mut self, value: T) {
        self.root = Self::insert_recursive(self.root.take(), value);
    }

    fn insert_recursive(node: Option<Box<TreeNode<T>>>, value: T) -> Option<Box<TreeNode<T>>> {
        match node {
            None => Some(Box::new(TreeNode::new(value))),
            Some(mut node) => {
                if value <= node.value {
                    node.left = Self::insert_recursive(node.left.take(), value);
                } else {
                    node.right = Self::insert_recursive(node.right.take(), value);
                }
                Some(node)
            }
        }
    }

    /// Search for a value in the BST
    pub fn search(&self, value: &T) -> bool {
        Self::search_recursive(&self.root, value)
    }

    fn search_recursive(node: &Option<Box<TreeNode<T>>>, value: &T) -> bool {
        match node {
            None => false,
            Some(node) => {
                if *value == node.value {
                    true
                } else if *value < node.value {
                    Self::search_recursive(&node.left, value)
                } else {
                    Self::search_recursive(&node.right, value)
                }
            }
        }
    }

    /// In-order traversal of the BST
    pub fn inorder(&self) -> Vec<&T> {
        let mut result = Vec::new();
        Self::inorder_recursive(&self.root, &mut result);
        result
    }

    fn inorder_recursive<'a>(node: &'a Option<Box<TreeNode<T>>>, result: &mut Vec<&'a T>) {
        if let Some(node) = node {
            Self::inorder_recursive(&node.left, result);
            result.push(&node.value);
            Self::inorder_recursive(&node.right, result);
        }
    }

    /// Pre-order traversal of the BST
    pub fn preorder(&self) -> Vec<&T> {
        let mut result = Vec::new();
        Self::preorder_recursive(&self.root, &mut result);
        result
    }

    fn preorder_recursive<'a>(node: &'a Option<Box<TreeNode<T>>>, result: &mut Vec<&'a T>) {
        if let Some(node) = node {
            result.push(&node.value);
            Self::preorder_recursive(&node.left, result);
            Self::preorder_recursive(&node.right, result);
        }
    }

    /// Post-order traversal of the BST
    pub fn postorder(&self) -> Vec<&T> {
        let mut result = Vec::new();
        Self::postorder_recursive(&self.root, &mut result);
        result
    }

    fn postorder_recursive<'a>(node: &'a Option<Box<TreeNode<T>>>, result: &mut Vec<&'a T>) {
        if let Some(node) = node {
            Self::postorder_recursive(&node.left, result);
            Self::postorder_recursive(&node.right, result);
            result.push(&node.value);
        }
    }

    /// Get the height of the tree
    pub fn height(&self) -> usize {
        Self::height_recursive(&self.root)
    }

    fn height_recursive(node: &Option<Box<TreeNode<T>>>) -> usize {
        match node {
            None => 0,
            Some(node) => {
                1 + std::cmp::max(
                    Self::height_recursive(&node.left),
                    Self::height_recursive(&node.right),
                )
            }
        }
    }

    /// Get a reference to the root node
    pub fn root(&self) -> &Option<Box<TreeNode<T>>> {
        &self.root
    }

    /// Serialize the tree to a string representation
    pub fn serialize(&self) -> String
    where
        T: fmt::Display,
    {
        Self::serialize_node(&self.root, 0)
    }

    fn serialize_node(node: &Option<Box<TreeNode<T>>>, depth: usize) -> String
    where
        T: fmt::Display,
    {
        match node {
            None => "null".to_string(),
            Some(node) => {
                let indent = "  ".repeat(depth);
                let mut result = format!("{}node: {}\n", indent, node.value);

                if node.left.is_some() || node.right.is_some() {
                    result.push_str(&format!("{indent}left:\n"));
                    result.push_str(&Self::serialize_node(&node.left, depth + 1));
                    result.push_str(&format!("{indent}right:\n"));
                    result.push_str(&Self::serialize_node(&node.right, depth + 1));
                }

                result
            }
        }
    }

    /// Visualize the tree structure
    pub fn visualize(&self) -> String
    where
        T: fmt::Display,
    {
        if self.root.is_none() {
            return "Empty tree".to_string();
        }

        Self::visualize_node(&self.root, "", true)
    }

    fn visualize_node(node: &Option<Box<TreeNode<T>>>, prefix: &str, is_last: bool) -> String
    where
        T: fmt::Display,
    {
        match node {
            None => String::new(),
            Some(node) => {
                let mut result = String::new();
                let connector = if is_last { "└── " } else { "├── " };
                result.push_str(&format!("{}{}{}\n", prefix, connector, node.value));

                let extension = if is_last { "    " } else { "│   " };
                let new_prefix = format!("{prefix}{extension}");

                if node.left.is_some() || node.right.is_some() {
                    if node.right.is_some() {
                        result.push_str(&Self::visualize_node(
                            &node.right,
                            &new_prefix,
                            node.left.is_none(),
                        ));
                    }
                    if node.left.is_some() {
                        result.push_str(&Self::visualize_node(&node.left, &new_prefix, true));
                    }
                }

                result
            }
        }
    }

    /// Compare two trees for structural equality
    pub fn structural_equals(&self, other: &Self) -> bool
    where
        T: PartialEq,
    {
        Self::nodes_equal(&self.root, &other.root)
    }

    fn nodes_equal(node1: &Option<Box<TreeNode<T>>>, node2: &Option<Box<TreeNode<T>>>) -> bool
    where
        T: PartialEq,
    {
        match (node1, node2) {
            (None, None) => true,
            (Some(n1), Some(n2)) => {
                n1.value == n2.value
                    && Self::nodes_equal(&n1.left, &n2.left)
                    && Self::nodes_equal(&n1.right, &n2.right)
            }
            _ => false,
        }
    }

    /// Compare tree structures (ignoring values)
    pub fn same_structure(&self, other: &Self) -> bool {
        Self::same_structure_nodes(&self.root, &other.root)
    }

    fn same_structure_nodes(
        node1: &Option<Box<TreeNode<T>>>,
        node2: &Option<Box<TreeNode<T>>>,
    ) -> bool {
        match (node1, node2) {
            (None, None) => true,
            (Some(n1), Some(n2)) => {
                Self::same_structure_nodes(&n1.left, &n2.left)
                    && Self::same_structure_nodes(&n1.right, &n2.right)
            }
            _ => false,
        }
    }

    /// Get tree statistics
    pub fn statistics(&self) -> TreeStatistics {
        let mut stats = TreeStatistics::default();
        Self::collect_statistics(&self.root, &mut stats, 0);
        stats
    }

    fn collect_statistics(
        node: &Option<Box<TreeNode<T>>>,
        stats: &mut TreeStatistics,
        depth: usize,
    ) {
        match node {
            None => {
                stats.max_depth = stats.max_depth.max(depth);
            }
            Some(node) => {
                stats.node_count += 1;
                stats.max_depth = stats.max_depth.max(depth);

                let has_left = node.left.is_some();
                let has_right = node.right.is_some();

                match (has_left, has_right) {
                    (false, false) => stats.leaf_count += 1,
                    (true, false) | (false, true) => stats.internal_count += 1,
                    (true, true) => stats.internal_count += 1,
                }

                Self::collect_statistics(&node.left, stats, depth + 1);
                Self::collect_statistics(&node.right, stats, depth + 1);
            }
        }
    }

    /// Check if the tree is balanced (AVL property)
    pub fn is_balanced(&self) -> bool {
        Self::check_balance(&self.root).is_some()
    }

    fn check_balance(node: &Option<Box<TreeNode<T>>>) -> Option<usize> {
        match node {
            None => Some(0),
            Some(node) => {
                let left_height = Self::check_balance(&node.left)?;
                let right_height = Self::check_balance(&node.right)?;

                if left_height.abs_diff(right_height) <= 1 {
                    Some(1 + left_height.max(right_height))
                } else {
                    None
                }
            }
        }
    }
}

/// Tree statistics for analysis
#[derive(Debug, Default, Clone)]
pub struct TreeStatistics {
    pub node_count: usize,
    pub leaf_count: usize,
    pub internal_count: usize,
    pub max_depth: usize,
}

/// Trie statistics for analysis
#[derive(Debug, Default, Clone)]
pub struct TrieStatistics {
    pub node_count: usize,
    pub leaf_count: usize,
    pub internal_count: usize,
    pub word_count: usize,
    pub max_depth: usize,
    pub branch_factor: usize,
}

/// Trie (Prefix Tree) for string storage and retrieval
#[derive(Clone, Debug)]
pub struct Trie {
    children: HashMap<char, Trie>,
    is_end_of_word: bool,
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

impl Trie {
    /// Create a new empty trie
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end_of_word: false,
        }
    }

    /// Insert a word into the trie
    pub fn insert(&mut self, word: &str) {
        let mut current = self;
        for ch in word.chars() {
            current = current.children.entry(ch).or_default();
        }
        current.is_end_of_word = true;
    }

    /// Search for a word in the trie
    pub fn search(&self, word: &str) -> bool {
        let mut current = self;
        for ch in word.chars() {
            match current.children.get(&ch) {
                Some(node) => current = node,
                None => return false,
            }
        }
        current.is_end_of_word
    }

    /// Check if any word starts with the given prefix
    pub fn starts_with(&self, prefix: &str) -> bool {
        let mut current = self;
        for ch in prefix.chars() {
            match current.children.get(&ch) {
                Some(node) => current = node,
                None => return false,
            }
        }
        true
    }

    /// Get all words with the given prefix
    pub fn words_with_prefix(&self, prefix: &str) -> Vec<String> {
        let mut current = self;
        for ch in prefix.chars() {
            match current.children.get(&ch) {
                Some(node) => current = node,
                None => return Vec::new(),
            }
        }

        let mut words = Vec::new();
        current.collect_words(prefix.to_string(), &mut words);
        words
    }

    fn collect_words(&self, prefix: String, words: &mut Vec<String>) {
        if self.is_end_of_word {
            words.push(prefix.clone());
        }

        for (&ch, child) in &self.children {
            let mut new_prefix = prefix.clone();
            new_prefix.push(ch);
            child.collect_words(new_prefix, words);
        }
    }

    /// Get all words stored in the trie
    pub fn all_words(&self) -> Vec<String> {
        let mut words = Vec::new();
        self.collect_words(String::new(), &mut words);
        words
    }

    /// Get the number of words stored in the trie
    pub fn word_count(&self) -> usize {
        let mut count = 0;
        if self.is_end_of_word {
            count += 1;
        }
        for child in self.children.values() {
            count += child.word_count();
        }
        count
    }

    /// Serialize the trie to a string representation
    pub fn serialize(&self) -> String {
        let mut result = String::new();
        self.serialize_node("".to_string(), &mut result, 0);
        result
    }

    fn serialize_node(&self, prefix: String, result: &mut String, depth: usize) {
        let indent = "  ".repeat(depth);

        if self.is_end_of_word {
            result.push_str(&format!("{indent}word: {prefix}\n"));
        }

        for (&ch, child) in &self.children {
            let mut new_prefix = prefix.clone();
            new_prefix.push(ch);
            result.push_str(&format!("{indent}char: {ch} -> \n"));
            child.serialize_node(new_prefix, result, depth + 1);
        }
    }

    /// Visualize the trie structure
    pub fn visualize(&self) -> String {
        let mut result = String::from("Trie\n");
        self.visualize_node("", "", true, &mut result);
        result
    }

    fn visualize_node(&self, prefix: &str, char_prefix: &str, is_last: bool, result: &mut String) {
        let connector = if is_last { "└── " } else { "├── " };

        if self.is_end_of_word {
            result.push_str(&format!("{char_prefix}{connector}[{prefix}] ✓\n"));
        } else if !prefix.is_empty() {
            result.push_str(&format!("{char_prefix}{connector}[{prefix}]\n"));
        }

        let extension = if is_last { "    " } else { "│   " };
        let new_char_prefix = format!("{char_prefix}{extension}");

        let children: Vec<_> = self.children.iter().collect();
        for (i, (&ch, child)) in children.iter().enumerate() {
            let is_last_child = i == children.len() - 1;
            let mut new_prefix = prefix.to_string();
            new_prefix.push(ch);
            child.visualize_node(&new_prefix, &new_char_prefix, is_last_child, result);
        }
    }

    /// Compare two tries for structural equality
    pub fn structural_equals(&self, other: &Self) -> bool {
        if self.is_end_of_word != other.is_end_of_word {
            return false;
        }

        if self.children.len() != other.children.len() {
            return false;
        }

        for (&ch, child) in &self.children {
            match other.children.get(&ch) {
                Some(other_child) => {
                    if !child.structural_equals(other_child) {
                        return false;
                    }
                }
                None => return false,
            }
        }

        true
    }

    /// Check if this trie contains all words from another trie
    pub fn contains_trie(&self, other: &Self) -> bool {
        for word in other.all_words() {
            if !self.search(&word) {
                return false;
            }
        }
        true
    }

    /// Get trie statistics
    pub fn statistics(&self) -> TrieStatistics {
        let mut stats = TrieStatistics::default();
        self.collect_trie_statistics(&mut stats, 0);
        stats
    }

    fn collect_trie_statistics(&self, stats: &mut TrieStatistics, depth: usize) {
        stats.node_count += 1;
        stats.max_depth = stats.max_depth.max(depth);

        if self.is_end_of_word {
            stats.word_count += 1;
        }

        if self.children.is_empty() {
            stats.leaf_count += 1;
        } else {
            stats.internal_count += 1;
            stats.branch_factor = stats.branch_factor.max(self.children.len());
        }

        for child in self.children.values() {
            child.collect_trie_statistics(stats, depth + 1);
        }
    }

    /// Remove a word from the trie
    pub fn remove(&mut self, word: &str) -> bool {
        let chars: Vec<char> = word.chars().collect();
        self.remove_recursive(&chars, 0).0
    }

    fn remove_recursive(&mut self, chars: &[char], index: usize) -> (bool, bool) {
        if index == chars.len() {
            if self.is_end_of_word {
                self.is_end_of_word = false;
                return (true, self.children.is_empty());
            }
            return (false, false);
        }

        let ch = chars[index];
        if let Some(child) = self.children.get_mut(&ch) {
            let (word_existed, should_remove_child) = child.remove_recursive(chars, index + 1);

            if should_remove_child {
                self.children.remove(&ch);
            }

            let can_remove_self = !self.is_end_of_word && self.children.is_empty();
            return (word_existed, can_remove_self);
        }

        (false, false)
    }

    /// Get the longest common prefix of all words in the trie
    pub fn longest_common_prefix(&self) -> String {
        let mut prefix = String::new();
        let mut current = self;

        while current.children.len() == 1 && !current.is_end_of_word {
            let (&ch, child) = current.children.iter().next().unwrap();
            prefix.push(ch);
            current = child;
        }

        prefix
    }
}

// ===== MEMORY-EFFICIENT DATA STRUCTURES =====

/// Ring buffer for efficient circular storage
#[derive(Clone, Debug)]
pub struct RingBuffer<T> {
    data: Vec<Option<T>>,
    capacity: usize,
    head: usize,
    tail: usize,
    size: usize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![None; capacity],
            capacity,
            head: 0,
            tail: 0,
            size: 0,
        }
    }

    /// Push an element to the buffer
    pub fn push(&mut self, item: T) -> Option<T> {
        let old_item = if self.size == self.capacity {
            self.data[self.tail].take()
        } else {
            None
        };

        self.data[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.capacity;

        if self.size == self.capacity {
            self.head = (self.head + 1) % self.capacity;
        } else {
            self.size += 1;
        }

        old_item
    }

    /// Pop the oldest element from the buffer
    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }

        let item = self.data[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.size -= 1;
        item
    }

    /// Get the current size of the buffer
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Get iterator over elements in insertion order
    pub fn iter(&self) -> RingBufferIter<'_, T> {
        RingBufferIter {
            buffer: self,
            index: 0,
        }
    }
}

/// Iterator for RingBuffer
pub struct RingBufferIter<'a, T> {
    buffer: &'a RingBuffer<T>,
    index: usize,
}

impl<'a, T: Clone> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.size {
            return None;
        }

        let actual_index = (self.buffer.head + self.index) % self.buffer.capacity;
        self.index += 1;

        self.buffer.data[actual_index].as_ref()
    }
}

/// Cache-friendly matrix storage with block layout
#[derive(Clone, Debug)]
pub struct BlockMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    block_size: usize,
}

impl<T: Clone + Zero> BlockMatrix<T> {
    /// Create a new block matrix with specified block size
    pub fn new(rows: usize, cols: usize, block_size: usize) -> Self {
        let total_size = rows * cols;
        Self {
            data: vec![T::zero(); total_size],
            rows,
            cols,
            block_size,
        }
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> UtilsResult<&T> {
        if row >= self.rows || col >= self.cols {
            return Err(UtilsError::InvalidParameter(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            )));
        }

        let index = self.block_index(row, col);
        Ok(&self.data[index])
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: T) -> UtilsResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(UtilsError::InvalidParameter(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            )));
        }

        let index = self.block_index(row, col);
        self.data[index] = value;
        Ok(())
    }

    fn block_index(&self, row: usize, col: usize) -> usize {
        let block_row = row / self.block_size;
        let block_col = col / self.block_size;
        let in_block_row = row % self.block_size;
        let in_block_col = col % self.block_size;

        let blocks_per_row = (self.cols + self.block_size - 1) / self.block_size;
        let block_index = block_row * blocks_per_row + block_col;
        let block_start = block_index * self.block_size * self.block_size;

        block_start + in_block_row * self.block_size + in_block_col
    }

    /// Get dimensions
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

// ===== CONCURRENT DATA STRUCTURES =====

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Thread-safe wrapper around HashMap
#[derive(Clone, Debug)]
pub struct ConcurrentHashMap<K, V> {
    inner: Arc<RwLock<HashMap<K, V>>>,
}

impl<K: Clone + Eq + Hash, V: Clone> Default for ConcurrentHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Clone + Eq + Hash, V: Clone> ConcurrentHashMap<K, V> {
    /// Create a new concurrent hashmap
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert a key-value pair
    pub fn insert(&self, key: K, value: V) -> UtilsResult<Option<V>> {
        let mut map = self
            .inner
            .write()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(map.insert(key, value))
    }

    /// Get a value by key
    pub fn get(&self, key: &K) -> UtilsResult<Option<V>> {
        let map = self
            .inner
            .read()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(map.get(key).cloned())
    }

    /// Remove a key-value pair
    pub fn remove(&self, key: &K) -> UtilsResult<Option<V>> {
        let mut map = self
            .inner
            .write()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(map.remove(key))
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> UtilsResult<bool> {
        let map = self
            .inner
            .read()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(map.contains_key(key))
    }

    /// Get the number of entries
    pub fn len(&self) -> UtilsResult<usize> {
        let map = self
            .inner
            .read()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(map.len())
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> UtilsResult<bool> {
        let map = self
            .inner
            .read()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(map.is_empty())
    }
}

/// Thread-safe ring buffer
#[derive(Clone, Debug)]
pub struct ConcurrentRingBuffer<T> {
    inner: Arc<Mutex<RingBuffer<T>>>,
}

impl<T: Clone> ConcurrentRingBuffer<T> {
    /// Create a new concurrent ring buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RingBuffer::new(capacity))),
        }
    }

    /// Push an element to the buffer
    pub fn push(&self, item: T) -> UtilsResult<Option<T>> {
        let mut buffer = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(buffer.push(item))
    }

    /// Pop the oldest element from the buffer
    pub fn pop(&self) -> UtilsResult<Option<T>> {
        let mut buffer = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(buffer.pop())
    }

    /// Get the current size of the buffer
    pub fn len(&self) -> UtilsResult<usize> {
        let buffer = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(buffer.len())
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> UtilsResult<bool> {
        let buffer = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(buffer.is_empty())
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> UtilsResult<bool> {
        let buffer = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(buffer.is_full())
    }
}

/// Thread-safe queue using VecDeque
#[derive(Clone, Debug)]
pub struct ConcurrentQueue<T> {
    inner: Arc<Mutex<VecDeque<T>>>,
}

impl<T: Clone> Default for ConcurrentQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> ConcurrentQueue<T> {
    /// Create a new concurrent queue
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Push an element to the back of the queue
    pub fn push_back(&self, item: T) -> UtilsResult<()> {
        let mut queue = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        queue.push_back(item);
        Ok(())
    }

    /// Push an element to the front of the queue
    pub fn push_front(&self, item: T) -> UtilsResult<()> {
        let mut queue = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        queue.push_front(item);
        Ok(())
    }

    /// Pop an element from the front of the queue
    pub fn pop_front(&self) -> UtilsResult<Option<T>> {
        let mut queue = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(queue.pop_front())
    }

    /// Pop an element from the back of the queue
    pub fn pop_back(&self) -> UtilsResult<Option<T>> {
        let mut queue = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(queue.pop_back())
    }

    /// Get the current size of the queue
    pub fn len(&self) -> UtilsResult<usize> {
        let queue = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(queue.len())
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> UtilsResult<bool> {
        let queue = self
            .inner
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(queue.is_empty())
    }
}

/// Atomic counter for thread-safe counting operations
#[derive(Clone, Debug)]
pub struct AtomicCounter {
    value: Arc<AtomicUsize>,
}

impl AtomicCounter {
    /// Create a new atomic counter
    pub fn new(initial: usize) -> Self {
        Self {
            value: Arc::new(AtomicUsize::new(initial)),
        }
    }

    /// Increment the counter and return the new value
    pub fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Decrement the counter and return the new value
    pub fn decrement(&self) -> usize {
        self.value.fetch_sub(1, Ordering::SeqCst).saturating_sub(1)
    }

    /// Add to the counter and return the new value
    pub fn add(&self, val: usize) -> usize {
        self.value.fetch_add(val, Ordering::SeqCst) + val
    }

    /// Subtract from the counter and return the new value
    pub fn sub(&self, val: usize) -> usize {
        self.value
            .fetch_sub(val, Ordering::SeqCst)
            .saturating_sub(val)
    }

    /// Get the current value
    pub fn get(&self) -> usize {
        self.value.load(Ordering::SeqCst)
    }

    /// Set the value
    pub fn set(&self, val: usize) {
        self.value.store(val, Ordering::SeqCst);
    }

    /// Compare and swap operation
    pub fn compare_and_swap(&self, current: usize, new: usize) -> usize {
        match self
            .value
            .compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(prev) => prev,
            Err(prev) => prev,
        }
    }
}

/// Thread-safe work queue for task distribution
#[derive(Clone, Debug)]
pub struct WorkQueue<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    active_workers: Arc<AtomicUsize>,
}

impl<T: Clone> Default for WorkQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> WorkQueue<T> {
    /// Create a new work queue
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            active_workers: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Add work to the queue
    pub fn add_work(&self, work: T) -> UtilsResult<()> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        queue.push_back(work);
        Ok(())
    }

    /// Get work from the queue
    pub fn get_work(&self) -> UtilsResult<Option<T>> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(queue.pop_front())
    }

    /// Register a worker as active
    pub fn register_worker(&self) {
        self.active_workers.fetch_add(1, Ordering::SeqCst);
    }

    /// Unregister a worker
    pub fn unregister_worker(&self) {
        self.active_workers.fetch_sub(1, Ordering::SeqCst);
    }

    /// Get number of active workers
    pub fn active_worker_count(&self) -> usize {
        self.active_workers.load(Ordering::SeqCst)
    }

    /// Check if there's work available
    pub fn has_work(&self) -> UtilsResult<bool> {
        let queue = self
            .queue
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(!queue.is_empty())
    }

    /// Get the current queue size
    pub fn queue_size(&self) -> UtilsResult<usize> {
        let queue = self
            .queue
            .lock()
            .map_err(|_| UtilsError::InvalidParameter("Lock poisoned".to_string()))?;
        Ok(queue.len())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_basic_operations() {
        let mut graph = Graph::new();

        let a = graph.add_vertex("A");
        let b = graph.add_vertex("B");
        let c = graph.add_vertex("C");

        graph.add_edge(&"A", &"B").unwrap();
        graph.add_edge(&"B", &"C").unwrap();
        graph.add_edge(&"C", &"A").unwrap();

        assert_eq!(graph.num_vertices(), 3);
        assert_eq!(graph.num_edges(), 3);

        let neighbors = graph.neighbors(&"A").unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], &"B");
    }

    #[test]
    fn test_graph_bfs() {
        let mut graph = Graph::new();

        graph.add_vertex("A");
        graph.add_vertex("B");
        graph.add_vertex("C");
        graph.add_vertex("D");

        graph.add_edge(&"A", &"B").unwrap();
        graph.add_edge(&"A", &"C").unwrap();
        graph.add_edge(&"B", &"D").unwrap();

        let bfs_result = graph.bfs(&"A").unwrap();
        assert_eq!(bfs_result[0], &"A");
        assert!(bfs_result.contains(&&"B"));
        assert!(bfs_result.contains(&&"C"));
        assert!(bfs_result.contains(&&"D"));
    }

    #[test]
    fn test_graph_has_cycle() {
        let mut graph = Graph::new();

        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);

        graph.add_edge(&1, &2).unwrap();
        graph.add_edge(&2, &3).unwrap();
        assert!(!graph.has_cycle());

        graph.add_edge(&3, &1).unwrap();
        assert!(graph.has_cycle());
    }

    #[test]
    fn test_binary_search_tree() {
        let mut bst = BinarySearchTree::new();

        bst.insert(5);
        bst.insert(3);
        bst.insert(7);
        bst.insert(1);
        bst.insert(9);

        assert!(bst.search(&5));
        assert!(bst.search(&1));
        assert!(!bst.search(&6));

        let inorder = bst.inorder();
        assert_eq!(inorder, vec![&1, &3, &5, &7, &9]);

        assert_eq!(bst.height(), 3);
    }

    #[test]
    fn test_trie() {
        let mut trie = Trie::new();

        trie.insert("cat");
        trie.insert("car");
        trie.insert("card");
        trie.insert("care");
        trie.insert("careful");

        assert!(trie.search("cat"));
        assert!(trie.search("car"));
        assert!(!trie.search("ca"));
        assert!(trie.starts_with("ca"));

        let words = trie.words_with_prefix("car");
        assert!(words.contains(&"car".to_string()));
        assert!(words.contains(&"card".to_string()));
        assert!(words.contains(&"care".to_string()));
        assert!(words.contains(&"careful".to_string()));
        assert!(!words.contains(&"cat".to_string()));
    }

    #[test]
    fn test_tree_serialization_visualization() {
        let mut bst = BinarySearchTree::new();
        bst.insert(5);
        bst.insert(3);
        bst.insert(7);
        bst.insert(1);

        let serialized = bst.serialize();
        assert!(serialized.contains("node: 5"));
        assert!(serialized.contains("node: 3"));

        let visualized = bst.visualize();
        assert!(visualized.contains("5"));
        assert!(visualized.contains("3"));
        assert!(visualized.contains("7"));
    }

    #[test]
    fn test_tree_comparison() {
        let mut bst1 = BinarySearchTree::new();
        bst1.insert(5);
        bst1.insert(3);
        bst1.insert(7);

        let mut bst2 = BinarySearchTree::new();
        bst2.insert(5);
        bst2.insert(3);
        bst2.insert(7);

        let mut bst3 = BinarySearchTree::new();
        bst3.insert(5);
        bst3.insert(2);
        bst3.insert(7);

        assert!(bst1.structural_equals(&bst2));
        assert!(!bst1.structural_equals(&bst3));
        assert!(bst1.same_structure(&bst2));
    }

    #[test]
    fn test_tree_statistics() {
        let mut bst = BinarySearchTree::new();
        bst.insert(5);
        bst.insert(3);
        bst.insert(7);
        bst.insert(1);
        bst.insert(9);

        let stats = bst.statistics();
        assert_eq!(stats.node_count, 5);
        assert_eq!(stats.leaf_count, 2); // 1 and 9 are leaves
        assert_eq!(stats.internal_count, 3); // 5, 3, 7 are internal
        assert!(stats.max_depth >= 2);

        assert!(bst.is_balanced());
    }

    #[test]
    fn test_trie_serialization_visualization() {
        let mut trie = Trie::new();
        trie.insert("cat");
        trie.insert("car");

        let serialized = trie.serialize();
        assert!(serialized.contains("word: cat"));
        assert!(serialized.contains("word: car"));

        let visualized = trie.visualize();
        assert!(visualized.contains("Trie"));
        assert!(visualized.contains("cat") || visualized.contains("car"));
    }

    #[test]
    fn test_trie_comparison() {
        let mut trie1 = Trie::new();
        trie1.insert("cat");
        trie1.insert("car");

        let mut trie2 = Trie::new();
        trie2.insert("cat");
        trie2.insert("car");

        let mut trie3 = Trie::new();
        trie3.insert("dog");

        assert!(trie1.structural_equals(&trie2));
        assert!(!trie1.structural_equals(&trie3));
        assert!(trie1.contains_trie(&trie2));
        assert!(!trie1.contains_trie(&trie3));
    }

    #[test]
    fn test_trie_statistics() {
        let mut trie = Trie::new();
        trie.insert("cat");
        trie.insert("car");
        trie.insert("care");

        let stats = trie.statistics();
        assert_eq!(stats.word_count, 3);
        assert!(stats.node_count >= 3);
        assert!(stats.max_depth >= 3);
        assert!(stats.branch_factor >= 1);
    }

    #[test]
    fn test_trie_removal() {
        let mut trie = Trie::new();
        trie.insert("cat");
        trie.insert("car");
        trie.insert("card");

        assert!(trie.search("car"));
        assert!(trie.remove("car"));
        assert!(!trie.search("car"));
        assert!(trie.search("card"));
        assert!(trie.search("cat"));
    }

    #[test]
    fn test_trie_longest_common_prefix() {
        let mut trie = Trie::new();
        trie.insert("preprocessing");
        trie.insert("preprocess");

        let prefix = trie.longest_common_prefix();
        assert!(prefix.starts_with("pre"));
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = RingBuffer::new(3);

        assert!(buffer.is_empty());
        assert!(!buffer.is_full());

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert!(buffer.is_full());
        assert_eq!(buffer.len(), 3);

        let old = buffer.push(4);
        assert_eq!(old, Some(1));

        let values: Vec<&i32> = buffer.iter().collect();
        assert_eq!(values, vec![&2, &3, &4]);

        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_block_matrix() {
        let mut matrix = BlockMatrix::new(4, 4, 2);

        matrix.set(0, 0, 1).unwrap();
        matrix.set(1, 1, 2).unwrap();
        matrix.set(2, 2, 3).unwrap();
        matrix.set(3, 3, 4).unwrap();

        assert_eq!(*matrix.get(0, 0).unwrap(), 1);
        assert_eq!(*matrix.get(1, 1).unwrap(), 2);
        assert_eq!(*matrix.get(2, 2).unwrap(), 3);
        assert_eq!(*matrix.get(3, 3).unwrap(), 4);

        assert_eq!(matrix.dim(), (4, 4));

        assert!(matrix.set(5, 5, 1).is_err());
        assert!(matrix.get(5, 5).is_err());
    }

    #[test]
    fn test_weighted_graph_mst() {
        let mut graph = WeightedGraph::new();

        graph.add_vertex("A");
        graph.add_vertex("B");
        graph.add_vertex("C");
        graph.add_vertex("D");

        graph.add_edge(&"A", &"B", 1).unwrap();
        graph.add_edge(&"B", &"A", 1).unwrap();
        graph.add_edge(&"B", &"C", 2).unwrap();
        graph.add_edge(&"C", &"B", 2).unwrap();
        graph.add_edge(&"C", &"D", 3).unwrap();
        graph.add_edge(&"D", &"C", 3).unwrap();
        graph.add_edge(&"A", &"D", 4).unwrap();
        graph.add_edge(&"D", &"A", 4).unwrap();

        let mst = graph.minimum_spanning_tree().unwrap();

        // MST should have 3 edges for 4 vertices
        assert!(mst.len() <= 3);

        // Check that all edges have reasonable weights
        for (_, _, weight) in &mst {
            assert!(*weight >= 1 && *weight <= 4);
        }
    }

    #[test]
    fn test_concurrent_hashmap() {
        let map = ConcurrentHashMap::new();

        assert!(map.insert("key1".to_string(), 1).unwrap().is_none());
        assert_eq!(map.get(&"key1".to_string()).unwrap(), Some(1));
        assert!(map.contains_key(&"key1".to_string()).unwrap());
        assert_eq!(map.len().unwrap(), 1);

        assert_eq!(map.insert("key1".to_string(), 2).unwrap(), Some(1));
        assert_eq!(map.get(&"key1".to_string()).unwrap(), Some(2));

        assert_eq!(map.remove(&"key1".to_string()).unwrap(), Some(2));
        assert_eq!(map.get(&"key1".to_string()).unwrap(), None);
        assert!(map.is_empty().unwrap());
    }

    #[test]
    fn test_concurrent_ring_buffer() {
        let buffer = ConcurrentRingBuffer::new(3);

        assert!(buffer.is_empty().unwrap());
        assert!(!buffer.is_full().unwrap());

        assert!(buffer.push(1).unwrap().is_none());
        assert!(buffer.push(2).unwrap().is_none());
        assert!(buffer.push(3).unwrap().is_none());

        assert!(buffer.is_full().unwrap());
        assert_eq!(buffer.len().unwrap(), 3);

        let old = buffer.push(4).unwrap();
        assert_eq!(old, Some(1));

        assert_eq!(buffer.pop().unwrap(), Some(2));
        assert_eq!(buffer.len().unwrap(), 2);
    }

    #[test]
    fn test_concurrent_queue() {
        let queue = ConcurrentQueue::new();

        assert!(queue.is_empty().unwrap());

        queue.push_back(1).unwrap();
        queue.push_back(2).unwrap();
        queue.push_front(0).unwrap();

        assert_eq!(queue.len().unwrap(), 3);

        assert_eq!(queue.pop_front().unwrap(), Some(0));
        assert_eq!(queue.pop_back().unwrap(), Some(2));
        assert_eq!(queue.pop_front().unwrap(), Some(1));

        assert!(queue.is_empty().unwrap());
    }

    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new(10);

        assert_eq!(counter.get(), 10);
        assert_eq!(counter.increment(), 11);
        assert_eq!(counter.decrement(), 10);
        assert_eq!(counter.add(5), 15);
        assert_eq!(counter.sub(3), 12);

        counter.set(100);
        assert_eq!(counter.get(), 100);

        assert_eq!(counter.compare_and_swap(100, 200), 100);
        assert_eq!(counter.get(), 200);
        assert_eq!(counter.compare_and_swap(100, 300), 200); // Should not change
        assert_eq!(counter.get(), 200);
    }

    #[test]
    fn test_work_queue() {
        let queue = WorkQueue::new();

        assert!(!queue.has_work().unwrap());
        assert_eq!(queue.queue_size().unwrap(), 0);
        assert_eq!(queue.active_worker_count(), 0);

        queue.add_work("task1".to_string()).unwrap();
        queue.add_work("task2".to_string()).unwrap();

        assert!(queue.has_work().unwrap());
        assert_eq!(queue.queue_size().unwrap(), 2);

        queue.register_worker();
        queue.register_worker();
        assert_eq!(queue.active_worker_count(), 2);

        assert_eq!(queue.get_work().unwrap(), Some("task1".to_string()));
        assert_eq!(queue.get_work().unwrap(), Some("task2".to_string()));
        assert_eq!(queue.get_work().unwrap(), None);

        queue.unregister_worker();
        assert_eq!(queue.active_worker_count(), 1);
    }

    #[test]
    fn test_graph_serialization() {
        let mut graph = Graph::new();

        graph.add_vertex("A");
        graph.add_vertex("B");
        graph.add_vertex("C");

        graph.add_edge(&"A", &"B").unwrap();
        graph.add_edge(&"B", &"C").unwrap();
        graph.add_edge(&"C", &"A").unwrap();

        let serialized = graph.serialize();
        assert!(serialized.contains("Graph {"));
        assert!(serialized.contains("vertices: 3 nodes"));
        assert!(serialized.contains("edges: 3 connections"));
        assert!(serialized.contains("A: [B]"));
        assert!(serialized.contains("B: [C]"));
        assert!(serialized.contains("C: [A]"));
    }

    #[test]
    fn test_graph_visualization() {
        let mut graph = Graph::new();

        graph.add_vertex("A");
        graph.add_vertex("B");

        graph.add_edge(&"A", &"B").unwrap();

        let visualized = graph.visualize();
        assert!(visualized.contains("Graph Visualization:"));
        assert!(visualized.contains("Vertices: 2"));
        assert!(visualized.contains("Edges: 1"));
        assert!(visualized.contains("Has cycle: false"));
        assert!(visualized.contains("A -> [B]"));
        assert!(visualized.contains("B -> []"));
    }

    #[test]
    fn test_graph_structural_equals() {
        let mut graph1 = Graph::new();
        graph1.add_vertex("A");
        graph1.add_vertex("B");
        graph1.add_edge(&"A", &"B").unwrap();

        let mut graph2 = Graph::new();
        graph2.add_vertex("A");
        graph2.add_vertex("B");
        graph2.add_edge(&"A", &"B").unwrap();

        let mut graph3 = Graph::new();
        graph3.add_vertex("A");
        graph3.add_vertex("C");
        graph3.add_edge(&"A", &"C").unwrap();

        assert!(graph1.structural_equals(&graph2));
        assert!(!graph1.structural_equals(&graph3));
    }

    #[test]
    fn test_weighted_graph_serialization() {
        let mut graph = WeightedGraph::new();

        graph.add_vertex("A");
        graph.add_vertex("B");
        graph.add_vertex("C");

        graph.add_edge(&"A", &"B", 1.5).unwrap();
        graph.add_edge(&"B", &"C", 2.0).unwrap();

        let serialized = graph.serialize();
        assert!(serialized.contains("WeightedGraph {"));
        assert!(serialized.contains("vertices: 3 nodes"));
        assert!(serialized.contains("edges: 2 weighted connections"));
        assert!(serialized.contains("A: [B:1.5]"));
        assert!(serialized.contains("B: [C:2]"));
        assert!(serialized.contains("C: []"));
    }

    #[test]
    fn test_weighted_graph_visualization() {
        let mut graph = WeightedGraph::new();

        graph.add_vertex("A");
        graph.add_vertex("B");

        graph.add_edge(&"A", &"B", 10).unwrap();

        let visualized = graph.visualize();
        assert!(visualized.contains("Weighted Graph Visualization:"));
        assert!(visualized.contains("Vertices: 2"));
        assert!(visualized.contains("Weighted Edges: 1"));
        assert!(visualized.contains("A -> [B(w:10)]"));
        assert!(visualized.contains("B -> []"));
    }

    #[test]
    fn test_weighted_graph_structural_equals() {
        let mut graph1 = WeightedGraph::new();
        graph1.add_vertex("A");
        graph1.add_vertex("B");
        graph1.add_edge(&"A", &"B", 5).unwrap();

        let mut graph2 = WeightedGraph::new();
        graph2.add_vertex("A");
        graph2.add_vertex("B");
        graph2.add_edge(&"A", &"B", 5).unwrap();

        let mut graph3 = WeightedGraph::new();
        graph3.add_vertex("A");
        graph3.add_vertex("B");
        graph3.add_edge(&"A", &"B", 10).unwrap();

        assert!(graph1.structural_equals(&graph2));
        assert!(!graph1.structural_equals(&graph3));
    }
}
