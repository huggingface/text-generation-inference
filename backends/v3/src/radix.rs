use std::collections::{hash_map::Entry, HashMap};

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

#[derive(Debug)]
pub struct TrieNode {
    children: HashMap<u32, TrieNode>,
    key: Vec<u32>,
    blocks: Vec<u32>,
    last_accessed: u64,
}

impl TrieNode {
    pub fn new(key: Vec<u32>, blocks: Vec<u32>, last_accessed: u64) -> Self {
        TrieNode {
            children: HashMap::new(),
            key,
            blocks,
            last_accessed,
        }
    }

    pub fn find(&self, key: &[u32], blocks: &mut Vec<u32>) {
        if let Some(child) = self.children.get(&key[0]) {
            let shared_prefix_len = child.key.shared_prefix_len(key);
            blocks.extend(&child.blocks[..shared_prefix_len]);

            let key = &key[shared_prefix_len..];
            if !key.is_empty() {
                child.find(key, blocks);
            }
        }
    }

    // Insert a prefix into the trie. Returns the length of the shared prefix.
    pub fn insert(&mut self, key: &[u32], blocks: &[u32]) -> usize {
        assert_eq!(key.len(), blocks.len());

        match self.children.entry(key[0]) {
            Entry::Occupied(entry) => {
                let child = entry.into_mut();
                let shared_prefix_len = child.key.shared_prefix_len(key);

                // We are done, the prefix is already in the trie.
                if shared_prefix_len == key.len() {
                    return shared_prefix_len;
                }

                // The node's prefix is a prefix of the insertion prefix.
                if shared_prefix_len == child.key.len() {
                    return shared_prefix_len
                        + child.insert(&key[shared_prefix_len..], &blocks[shared_prefix_len..]);
                }

                // The node's prefix and the insertion prefix only match partially,
                // split the node to just contain the matching part. Then insert the
                // remainder of the prefix into the node again.
                child.split(shared_prefix_len);
                let key = &key[shared_prefix_len..];
                let blocks = &blocks[shared_prefix_len..];
                child.insert(key, blocks)
            }
            Entry::Vacant(entry) => {
                let child = TrieNode::new(key.to_vec(), blocks.to_vec(), self.last_accessed);
                entry.insert(child);
                return key.len();
            }
        }

        //node.last_accessed = last_accessed;
    }

    fn split(&mut self, prefix_len: usize) {
        let rest_key = self.key.split_off(prefix_len);
        let rest_blocks = self.blocks.split_off(prefix_len);

        self.children.insert(
            rest_key[0],
            TrieNode::new(rest_key, rest_blocks, self.last_accessed),
        );
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
        let mut root = super::TrieNode::new(vec![], vec![], 0);

        assert_eq!(root.insert(&[0, 1, 2], &[0, 1, 2]), 3);

        // Already exists.
        assert_eq!(root.insert(&[0, 1, 2], &[0, 1, 2]), 3);

        // Completely new at root-level
        assert_eq!(root.insert(&[1, 2, 3], &[1, 2, 3]), 3);

        // Contains full prefix, but longer.
        assert_eq!(root.insert(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]), 5);

        // Shares partial prefix, we need a split.
        assert_eq!(
            root.insert(&[0, 1, 2, 3, 5, 6, 7], &[0, 1, 2, 3, 5, 6, 7]),
            6
        );
    }

    #[test]
    fn prefix_get_returns_correct_blocks() {
        let mut root = super::TrieNode::new(vec![], vec![], 0);
        root.insert(&[0, 1, 2], &[0, 1, 2]);
        root.insert(&[1, 2, 3], &[1, 2, 3]);
        root.insert(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]);
        root.insert(&[0, 1, 2, 3, 5, 6, 7], &[0, 1, 2, 3, 5, 6, 7]);

        let mut blocks = Vec::new();
        root.find(&[0], &mut blocks);
        assert_eq!(blocks, vec![0]);

        blocks.clear();
        root.find(&[0, 1, 2], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2]);

        blocks.clear();
        root.find(&[1, 2, 3], &mut blocks);
        assert_eq!(blocks, vec![1, 2, 3]);

        blocks.clear();
        root.find(&[0, 1, 2, 3], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2, 3]);

        blocks.clear();
        root.find(&[0, 1, 2, 3, 4], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2, 3, 4]);

        blocks.clear();
        root.find(&[0, 1, 2, 3, 5], &mut blocks);
        assert_eq!(blocks, vec![0, 1, 2, 3, 5]);
    }
}
