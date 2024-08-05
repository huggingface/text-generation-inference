use std::collections::{BTreeSet, HashMap};

use slotmap::{DefaultKey, SlotMap};

// Radix trie that is heavily inspired by radix attention from sglang.
//
// The trie is optimized for prefix caching:
//
// - A normal radix trie stores discrete values. In this radix trie,
//   inserting *abc* with value *xyz* will also enable lookup for
//   *a* (*x*) and *ab* (*xy*).
// - As a result, every value is required to have the same length as
//   the key.
// - We store additional information in each node, such as last access
//   time and a reference count.

type NodeId = DefaultKey;

#[derive(Debug)]
pub struct RadixTrie {
    root: DefaultKey,
    leaves: BTreeSet<(u64, NodeId)>,
    nodes: SlotMap<NodeId, TrieNode>,
    time: u64,
}

impl RadixTrie {
    pub fn new() -> Self {
        let root = TrieNode::new(vec![], vec![], 0, None);
        let mut nodes = SlotMap::new();
        let root = nodes.insert(root);
        RadixTrie {
            leaves: BTreeSet::new(),
            nodes,
            root,
            time: 0,
        }
    }

    pub fn find(&mut self, key: &[u32], blocks: &mut Vec<u32>) -> NodeId {
        self.time += 1;
        self.find_(self.root, key, blocks)
    }

    fn find_(&mut self, mut node_id: NodeId, key: &[u32], blocks: &mut Vec<u32>) -> NodeId {
        let node = &self.nodes[node_id];

        if let Some(&child_id) = node.children.get(&key[0]) {
            self.update_access_time(child_id);
            let child = self.nodes.get(child_id).unwrap();
            let shared_prefix_len = child.key.shared_prefix_len(key);
            blocks.extend(&child.blocks[..shared_prefix_len]);

            let key = &key[shared_prefix_len..];
            if !key.is_empty() {
                node_id = self.find_(child_id, key, blocks);
            }
        }

        node_id
    }

    pub fn decref(&mut self, node_id: NodeId) {
        if node_id == self.root {
            return;
        }

        let node = self.nodes.get_mut(node_id).unwrap();
        assert!(node.ref_count > 0);
        node.ref_count -= 1;
        if node.ref_count == 0 {
            self.leaves.insert((node.last_accessed, node_id));
        }
    }

    /// Evict `n_blocks` from the trie.
    ///
    /// Returns the evicted blocks.
    pub fn evict(&mut self, n_blocks: usize) -> Vec<u32> {
        let mut evicted = Vec::new();

        while let Some((last_access, node_id)) = self.leaves.pop_first() {
            let blocks_needed = n_blocks - evicted.len();

            let node = self.nodes.get(node_id).unwrap();

            if blocks_needed >= node.blocks.len() {
                // We need to evict the whole node if we exhaust all its blocks.
                let node = self.remove_node(node_id);
                evicted.extend(node.blocks);

                if evicted.len() >= n_blocks {
                    break;
                }
            } else {
                // The node has more blocks than needed, so we'll just remove
                // what is needed.
                let node = self.nodes.get_mut(node_id).unwrap();
                node.key.truncate(node.blocks.len() - blocks_needed);
                evicted.extend(node.blocks.split_off(node.blocks.len() - blocks_needed));
                self.leaves.insert((last_access, node_id));
                break;
            }
        }

        evicted
    }

    pub fn incref(&mut self, node_id: NodeId) {
        if node_id == self.root {
            return;
        }

        let node = self.nodes.get_mut(node_id).unwrap();
        if node.ref_count == 0 {
            self.leaves.remove(&(node.last_accessed, node_id));
        }
        node.ref_count += 1;
    }

    pub fn insert(&mut self, key: &[u32], blocks: &[u32]) -> usize {
        self.time += 1;
        self.insert_(self.root, key, blocks)
    }

    fn insert_(&mut self, node_id: NodeId, key: &[u32], blocks: &[u32]) -> usize {
        assert_eq!(key.len(), blocks.len());

        if let Some(&child_id) = self.nodes[node_id].children.get(&key[0]) {
            self.update_access_time(child_id);
            let child = self.nodes.get_mut(child_id).unwrap();
            let shared_prefix_len = child.key.shared_prefix_len(key);

            // We are done, the prefix is already in the trie.
            if shared_prefix_len == key.len() {
                return shared_prefix_len;
            }

            // The node's prefix is a prefix of the insertion prefix.
            if shared_prefix_len == child.key.len() {
                return shared_prefix_len
                    + self.insert_(
                        child_id,
                        &key[shared_prefix_len..],
                        &blocks[shared_prefix_len..],
                    );
            }

            // The node's prefix and the insertion prefix only match partially,
            // split the node to just contain the matching part. Then insert the
            // remainder of the prefix into the node again
            let child_id = self.split_node(child_id, shared_prefix_len);
            let key = &key[shared_prefix_len..];
            let blocks = &blocks[shared_prefix_len..];
            self.insert_(child_id, key, blocks)
        } else {
            self.add_node(node_id, key, blocks);
            key.len()
        }
    }

    fn split_node(&mut self, node_id: NodeId, prefix_len: usize) -> NodeId {
        // We have to make the current node a child to ensure that its
        // properties and node id stay the same.

        let node = self.nodes.get_mut(node_id).unwrap();
        let mut parent_key = node.key.split_off(prefix_len);
        let mut parent_blocks = node.blocks.split_off(prefix_len);

        // Move first part of the prefix to the parent.
        std::mem::swap(&mut node.key, &mut parent_key);
        std::mem::swap(&mut node.blocks, &mut parent_blocks);

        let node_key = node.key[0];

        let grandparent_id = node.parent.unwrap();
        let parent_id = self.add_node(grandparent_id, parent_key, parent_blocks);
        self.add_node_to_parent(parent_id, node_key, node_id);

        let node = self.nodes.get_mut(node_id).unwrap();
        node.parent = Some(parent_id);

        parent_id
    }

    fn add_node(
        &mut self,
        parent_id: NodeId,
        key: impl Into<Vec<u32>>,
        blocks: impl Into<Vec<u32>>,
    ) -> NodeId {
        let key = key.into();
        let blocks = blocks.into();
        let first = key[0];

        let child = TrieNode::new(key, blocks, self.time, Some(parent_id));
        let child_id = self.nodes.insert(child);

        self.add_node_to_parent(parent_id, first, child_id);
        self.leaves.insert((self.time, child_id));

        child_id
    }

    fn add_node_to_parent(&mut self, parent_id: NodeId, first: u32, child_id: NodeId) {
        let parent = self.nodes.get_mut(parent_id).unwrap();
        if !parent.children.insert(first, child_id).is_some() {
            // Only increase reference count if child does not replace another child.
            self.incref(parent_id);
        }
    }

    fn remove_node(&mut self, node_id: NodeId) -> TrieNode {
        let node = self.nodes.remove(node_id).unwrap();
        let parent_id = node.parent.expect("Attempted to remove root node");
        let parent = self.nodes.get_mut(parent_id).unwrap();
        parent.children.remove(&node.key[0]);
        self.decref(parent_id);
        self.nodes.remove(node_id);
        node
    }

    fn update_access_time(&mut self, node_id: NodeId) {
        let node = self.nodes.get_mut(node_id).unwrap();

        // Update the ordered leaves set if the node is a leave.
        if self.leaves.remove(&(node.last_accessed, node_id)) {
            self.leaves.insert((self.time, node_id));
        }

        node.last_accessed = self.time;
    }

    pub fn print_debug(&self) {
        self.print_debug_(self.root, 0);
    }

    fn print_debug_(&self, node_id: NodeId, indent: usize) {
        let node = &self.nodes[node_id];
        eprintln!(
            "{}{:?}, key: {:?}, blocks: {:?}, ref_count: {}, last_accessed: {}, parent: {:?}, children: {:?}",
            " ".repeat(indent),
            node_id,
            node.key,
            node.blocks,
            node.ref_count,
            node.last_accessed,
            node.parent,
            node.children
        );
        for child_id in self.nodes[node_id].children.values() {
            self.print_debug_(*child_id, indent + 2);
        }
    }
}

#[derive(Debug)]
struct TrieNode {
    blocks: Vec<u32>,
    children: HashMap<u32, NodeId>,
    key: Vec<u32>,
    last_accessed: u64,
    parent: Option<NodeId>,
    ref_count: usize,
}

impl TrieNode {
    fn new(key: Vec<u32>, blocks: Vec<u32>, last_accessed: u64, parent: Option<NodeId>) -> Self {
        TrieNode {
            children: HashMap::new(),
            key,
            blocks,
            last_accessed,
            parent,
            ref_count: 0,
        }
    }
}

trait SharedPrefixLen {
    fn shared_prefix_len(&self, other: &Self) -> usize;
}

impl<T> SharedPrefixLen for [T]
where
    T: PartialEq,
{
    fn shared_prefix_len(&self, other: &Self) -> usize {
        self.iter().zip(other).take_while(|(a, b)| a == b).count()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn insertions_have_correct_prefix_len() {
        let mut trie = super::RadixTrie::new();

        assert_eq!(trie.insert(&[0, 1, 2], &[0, 1, 2]), 3);

        // Already exists.
        assert_eq!(trie.insert(&[0, 1, 2], &[0, 1, 2]), 3);

        // Completely new at root-level
        assert_eq!(trie.insert(&[1, 2, 3], &[1, 2, 3]), 3);

        // Contains full prefix, but longer.
        assert_eq!(trie.insert(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]), 5);

        // Shares partial prefix, we need a split.
        assert_eq!(
            trie.insert(&[0, 1, 2, 3, 5, 6, 7], &[0, 1, 2, 3, 5, 6, 7]),
            6
        );
    }

    #[test]
    fn prefix_get_returns_correct_blocks() {
        let mut trie = super::RadixTrie::new();
        trie.insert(&[0, 1, 2], &[0, 1, 2]);
        trie.insert(&[1, 2, 3], &[1, 2, 3]);
        trie.insert(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]);
        trie.insert(&[0, 1, 2, 3, 5, 6, 7], &[0, 1, 2, 3, 5, 6, 7]);

        let mut blocks = Vec::new();
        trie.find(&[0], &mut blocks);
        assert_eq!(blocks, vec![0]);

        blocks.clear();
        trie.find(&[0, 1, 2], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2]);

        blocks.clear();
        trie.find(&[1, 2, 3], &mut blocks);
        assert_eq!(blocks, vec![1, 2, 3]);

        blocks.clear();
        trie.find(&[0, 1, 2, 3], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2, 3]);

        blocks.clear();
        trie.find(&[0, 1, 2, 3, 4], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2, 3, 4]);

        blocks.clear();
        trie.find(&[0, 1, 2, 3, 5], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2, 3, 5]);
    }

    #[test]
    fn evict_removes_correct_blocks() {
        let mut trie = super::RadixTrie::new();
        trie.insert(&[0, 1, 2], &[0, 1, 2]);
        trie.insert(&[0, 1, 2, 3, 5, 6, 7], &[0, 1, 2, 3, 5, 6, 7]);
        trie.insert(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]);
        trie.insert(&[1, 2, 3], &[1, 2, 3]);

        let mut blocks = Vec::new();

        // Remove less than the leave blocks.
        assert_eq!(trie.evict(1), vec![7]);
        trie.find(&[0, 1, 2, 3, 5, 6, 7], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2, 3, 5, 6]);

        // Refresh other leaf.
        trie.find(&[0, 1, 2, 3, 4], &mut blocks);
        trie.find(&[1, 2, 3], &mut blocks);

        // Remove the leave blocks exactly.
        assert_eq!(trie.evict(2), vec![5, 6]);
        blocks.clear();
        trie.find(&[0, 1, 2, 3, 5, 6, 7], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2, 3]);

        trie.find(&[1, 2, 3], &mut blocks);

        // Remove more than the leave blocks.
        assert_eq!(trie.evict(3), vec![4, 3, 2]);
        blocks.clear();
        trie.find(&[0, 1, 2, 3, 4], &mut blocks);
        assert_eq!(blocks, vec![0, 1]);

        // Clear out the whole trie.
        assert_eq!(trie.evict(10), vec![1, 2, 3, 0, 1]);
    }
}
