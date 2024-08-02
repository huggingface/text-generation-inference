use std::collections::{hash_map::Entry, HashMap};

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

pub struct RadixTrie {
    root: DefaultKey,
    nodes: SlotMap<NodeId, TrieNode>,
    time: u64,
}

impl RadixTrie {
    pub fn new() -> Self {
        let root = TrieNode::new(vec![], vec![], 0, None);
        let mut nodes = SlotMap::new();
        let root = nodes.insert(root);
        RadixTrie {
            nodes,
            root,
            time: 0,
        }
    }

    pub fn find(&mut self, key: &[u32], blocks: &mut Vec<u32>) {
        self.time += 1;
        self.find_(self.root, key, blocks);
    }

    fn find_(&mut self, node_id: NodeId, key: &[u32], blocks: &mut Vec<u32>) {
        let node = &self.nodes[node_id];

        if let Some(&child_id) = node.children.get(&key[0]) {
            let child = self.nodes.get_mut(child_id).unwrap();
            child.last_accessed = self.time;
            let shared_prefix_len = child.key.shared_prefix_len(key);
            blocks.extend(&child.blocks[..shared_prefix_len]);

            let key = &key[shared_prefix_len..];
            if !key.is_empty() {
                self.find_(child_id, key, blocks);
            }
        }
    }

    pub fn insert(&mut self, key: &[u32], blocks: &[u32]) -> usize {
        self.time += 1;
        self.insert_(self.root, key, blocks)
    }

    fn insert_(&mut self, node_id: NodeId, key: &[u32], blocks: &[u32]) -> usize {
        assert_eq!(key.len(), blocks.len());

        if let Some(&child_id) = self.nodes[node_id].children.get(&key[0]) {
            let child = self.nodes.get_mut(child_id).unwrap();
            child.last_accessed = self.time;
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
            // remainder of the prefix into the node again.
            self.split(child_id, shared_prefix_len);
            let key = &key[shared_prefix_len..];
            let blocks = &blocks[shared_prefix_len..];
            self.insert_(child_id, key, blocks)
        } else {
            let child = TrieNode::new(key.to_vec(), blocks.to_vec(), self.time, Some(node_id));
            let child_id = self.nodes.insert(child);
            let node = self.nodes.get_mut(node_id).unwrap();
            node.children.insert(key[0], child_id);
            key.len()
        }
    }

    fn split(&mut self, node_id: NodeId, prefix_len: usize) {
        let node = self.nodes.get_mut(node_id).unwrap();

        let rest_key = node.key.split_off(prefix_len);
        let rest_blocks = node.blocks.split_off(prefix_len);
        let first = rest_key[0];

        let new_id = self.nodes.insert(TrieNode::new(
            rest_key,
            rest_blocks,
            self.time,
            Some(node_id),
        ));

        let node = self.nodes.get_mut(node_id).unwrap();
        node.children.insert(first, new_id);
    }
}

struct TrieNode {
    blocks: Vec<u32>,
    children: HashMap<u32, NodeId>,
    key: Vec<u32>,
    last_accessed: u64,
    parent: Option<NodeId>,
}

impl TrieNode {
    fn new(key: Vec<u32>, blocks: Vec<u32>, last_accessed: u64, parent: Option<NodeId>) -> Self {
        TrieNode {
            children: HashMap::new(),
            key,
            blocks,
            last_accessed,
            parent,
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
}
