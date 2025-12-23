//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::super::types::*;
    #[test]
    fn test_graph_basic_operations() {
        let mut graph = Graph::new();
        let _a = graph.add_vertex("A");
        let _b = graph.add_vertex("B");
        let _c = graph.add_vertex("C");
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
        assert_eq!(stats.leaf_count, 2);
        assert_eq!(stats.internal_count, 3);
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
        assert!(mst.len() <= 3);
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
        assert_eq!(counter.compare_and_swap(100, 300), 200);
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
