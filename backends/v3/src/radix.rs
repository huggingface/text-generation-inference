use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use slotmap::{DefaultKey, SlotMap};

use crate::block_allocator::{Allocator, BlockAllocation};

pub struct RadixAllocator {
    allocation_id: u64,

    allocations: HashMap<u64, RadixAllocation>,

    cache_blocks: RadixTrie,

    /// Blocks that are immediately available for allocation.
    free_blocks: Vec<u32>,
}

impl RadixAllocator {
    pub fn new(block_size: u32, n_blocks: u32, window_size: Option<u32>) -> Self {
        assert_eq!(
            block_size, 1,
            "Radix tree allocator only works with block_size=1, was: {}",
            block_size
        );
        if window_size.is_some() {
            unimplemented!("Window size not supported in the prefix-caching block allocator yet");
        }

        RadixAllocator {
            allocation_id: 0,
            allocations: HashMap::new(),
            cache_blocks: RadixTrie::new(),

            // Block 0 is reserved for health checks.
            free_blocks: (1..n_blocks).collect(),
        }
    }

    fn alloc_or_reclaim(&mut self, n_blocks_needed: usize) -> Option<Vec<u32>> {
        if self.free_blocks.len() < n_blocks_needed {
            // This is a bit annoying, we first extend the free list and then
            // split it off again below. This is because we need to put it on
            // the free list if we cannot allocate enough blocks. This is only
            // temporary, the trie needs to be able to report whether it can
            // allocate the requested amount. Just not implemented yet.
            self.free_blocks.extend(
                self.cache_blocks
                    .evict(n_blocks_needed - self.free_blocks.len()),
            );
        }

        if self.free_blocks.len() >= n_blocks_needed {
            Some(
                self.free_blocks
                    .split_off(self.free_blocks.len() - n_blocks_needed),
            )
        } else {
            None
        }
    }
}

impl Allocator for RadixAllocator {
    fn allocate(
        &mut self,
        tokens: u32,
        prefill_tokens: Option<Arc<Vec<u32>>>,
    ) -> Option<BlockAllocation> {
        let mut blocks = vec![];
        let prefix_node = if let Some(prefill_tokens) = prefill_tokens.as_ref() {
            let node_id = self
                .cache_blocks
                .find(prefill_tokens.as_slice(), &mut blocks);
            // Even if this allocation fails below, we need to increase he
            // refcount to ensure that the prefix that was found is not evicted.

            node_id
        } else {
            self.cache_blocks.root_id()
        };

        self.cache_blocks
            .incref(prefix_node)
            .expect("Failed to increment refcount");

        let prefix_len = blocks.len();
        let suffix_len = tokens - prefix_len as u32;

        match self.alloc_or_reclaim(suffix_len as usize) {
            Some(suffix_blocks) => blocks.extend(suffix_blocks),
            None => {
                self.cache_blocks
                    .decref(prefix_node)
                    .expect("Failed to decrement refcount");
                return None;
            }
        }

        // 1:1 mapping of blocks and slots.
        let slots = blocks.clone();

        let allocation = RadixAllocation {
            prefix_node,
            cached_prefix_len: prefix_len,
            prefill_tokens: prefill_tokens.clone(),
        };

        self.allocation_id += 1;
        self.allocations.insert(self.allocation_id, allocation);

        Some(BlockAllocation {
            allocation_id: self.allocation_id,
            block_allocator: None,
            blocks,
            slots,
            prefix_len: prefix_len as u32,
        })
    }

    fn free(&mut self, blocks: Vec<u32>, allocation_id: u64) {
        let allocation = match self.allocations.remove(&allocation_id) {
            Some(allocation) => allocation,
            None => unreachable!("Tried to free an unknown allocation."),
        };

        self.cache_blocks
            .decref(allocation.prefix_node)
            .expect("Failed to decrement refcount");

        if let Some(prefill_tokens) = allocation.prefill_tokens {
            let prefill_tokens = prefill_tokens.as_slice();

            // If there are prefill tokens that did not come from the cache,
            // add them to the cache.
            if prefill_tokens.len() > allocation.cached_prefix_len {
                let prefix_len = self
                    .cache_blocks
                    .insert(prefill_tokens, &blocks[..prefill_tokens.len()])
                    // Unwrap, failing is a programming error.
                    .expect("Failed to store prefill tokens");

                // We can have a prefill with the following structure:
                //
                // |---| From the prefix cache.
                // A B C D E F G
                //|--------| Found in the trie during insertion.
                //
                // This means that while processing this request there was a
                // partially overlapping request that had A..=E in its
                // prefill. In this case we need to free the blocks D E.
                self.free_blocks
                    .extend(&blocks[allocation.cached_prefix_len..prefix_len]);
            }

            // Free non-prefill blocks.
            self.free_blocks.extend(&blocks[prefill_tokens.len()..]);
        } else {
            self.free_blocks.extend(blocks);
        }
    }
}

struct RadixAllocation {
    prefix_node: NodeId,
    cached_prefix_len: usize,
    prefill_tokens: Option<Arc<Vec<u32>>>,
}

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
pub enum TrieError {
    InvalidNodeId,
    RefCountUnderflow,
    BlockTokenCountMismatch,
}

pub type NodeId = DefaultKey;

#[derive(Debug)]
pub struct RadixTrie {
    /// Identifier of the root nod.
    root: DefaultKey,

    /// Leave node identifiers ordered by increasing recency.
    leaves: BTreeSet<(u64, NodeId)>,

    /// All trie nodes.
    nodes: SlotMap<NodeId, TrieNode>,

    /// Time as a monotonically increating counter to avoid the system
    /// call that a real time lookup would require.
    time: u64,
}

impl RadixTrie {
    /// Construct a new radix trie.
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

    /// Find the prefix of the given tokens.
    ///
    /// The blocks corresponding to the part of the prefix that could be found
    /// are writteng to `blocks`. The number of blocks is in `0..=tokens.len()`.
    /// Returns the identifier of the trie node that contains the longest
    /// prefix. The node identifier can be used by callers to e.g. increase its
    /// reference count.
    ///
    /// Using this method will update the access time of the traversed nodes.
    pub fn find(&mut self, key: &[u32], blocks: &mut Vec<u32>) -> NodeId {
        self.time += 1;
        self.find_(self.root, key, blocks)
    }

    /// Find worker.
    fn find_(&mut self, mut node_id: NodeId, key: &[u32], blocks: &mut Vec<u32>) -> NodeId {
        let node = &self.nodes[node_id];

        if let Some(&child_id) = node.children.get(&key[0]) {
            self.update_access_time(child_id);
            let child = self.nodes.get(child_id).expect("Invalid child identifier");
            let shared_prefix_len = child.key.shared_prefix_len(key);
            blocks.extend(&child.blocks[..shared_prefix_len]);

            let key = &key[shared_prefix_len..];
            if !key.is_empty() {
                node_id = self.find_(child_id, key, blocks);
            }
        }

        node_id
    }

    /// Decrease the reference count of a node.
    pub fn decref(&mut self, node_id: NodeId) -> Result<(), TrieError> {
        // We don't care about refcounting for root, since it will never
        // be evicted.
        if node_id == self.root {
            return Ok(());
        }

        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or(TrieError::InvalidNodeId)?;
        if node.ref_count == 0 {
            return Err(TrieError::RefCountUnderflow);
        }

        node.ref_count -= 1;
        if node.ref_count == 0 {
            self.leaves.insert((node.last_accessed, node_id));
        }

        Ok(())
    }

    /// Increase the reference count of a node.
    pub fn incref(&mut self, node_id: NodeId) -> Result<(), TrieError> {
        if node_id == self.root {
            return Ok(());
        }

        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or(TrieError::InvalidNodeId)?;
        if node.ref_count == 0 {
            self.leaves.remove(&(node.last_accessed, node_id));
        }
        node.ref_count += 1;

        Ok(())
    }

    /// Evict `n_blocks` from the trie.
    ///
    /// Returns the evicted blocks. When the length is less than `n_blocks`,
    /// not enough blocks could beevicted.
    pub fn evict(&mut self, n_blocks: usize) -> Vec<u32> {
        // NOTE: we don't return Result here. If any of the unwrapping fails,
        // it's a programming error in the trie implementation, not a user
        // error caused by e.g. an invalid argument.

        // TODO: add some bookkeeping in the future to check whether we can
        // evict n_blocks and return `None` if we can't. We are now needlessly
        // evicting prefixes from the cache in such a case.
        let mut evicted = Vec::new();

        while let Some((last_access, node_id)) = self.leaves.pop_first() {
            let blocks_needed = n_blocks - evicted.len();

            let node = self.nodes.get(node_id).expect("Leave does not exist");
            if blocks_needed >= node.blocks.len() {
                // We need to evict the whole node if we need more blocks than it has.
                let node = self.remove_node(node_id);
                evicted.extend(node.blocks);

                if evicted.len() >= n_blocks {
                    break;
                }
            } else {
                // The node has more blocks than needed, so we'll just remove
                // the required number of blocks and leave the remaining blocks
                // untouched.
                let node = self.nodes.get_mut(node_id).expect("Leave does not exist");
                node.key.truncate(node.blocks.len() - blocks_needed);
                evicted.extend(node.blocks.split_off(node.blocks.len() - blocks_needed));
                self.leaves.insert((last_access, node_id));
                break;
            }
        }

        evicted
    }

    /// Insert a prefill along with its blocks.
    ///
    /// This method returns the length of the prefix that was already
    /// in the trie. E.g. if the length is 10, this means that for
    /// the first 10 elements of the tree **the blocks are not updated**.
    pub fn insert(&mut self, tokens: &[u32], blocks: &[u32]) -> Result<usize, TrieError> {
        self.time += 1;
        self.insert_(self.root, tokens, blocks)
    }

    /// Insertion worker.
    fn insert_(
        &mut self,
        node_id: NodeId,
        tokens: &[u32],
        blocks: &[u32],
    ) -> Result<usize, TrieError> {
        // TODO: in the future we may want to check that the blocks match for
        // the part of the prefix that is already in the trie to detect
        // mismatches.

        if tokens.len() != blocks.len() {
            return Err(TrieError::BlockTokenCountMismatch);
        }

        if let Some(&child_id) = self.nodes[node_id].children.get(&tokens[0]) {
            self.update_access_time(child_id);
            let child = self
                .nodes
                .get_mut(child_id)
                // Unwrap here, since failure is a bug.
                .expect("Child node does not exist");
            let shared_prefix_len = child.key.shared_prefix_len(tokens);

            // We are done, the prefix is already in the trie.
            if shared_prefix_len == tokens.len() {
                return Ok(shared_prefix_len);
            }

            // The node's prefix is a prefix of the insertion prefix.
            if shared_prefix_len == child.key.len() {
                return Ok(shared_prefix_len
                    + self.insert_(
                        child_id,
                        &tokens[shared_prefix_len..],
                        &blocks[shared_prefix_len..],
                    )?);
            }

            // The node's prefix and the insertion prefix only match partially,
            // split the node to just contain the matching part. Then insert the
            // remainder of the prefix into the node again
            let child_id = self.split_node(child_id, shared_prefix_len);
            let key = &tokens[shared_prefix_len..];
            let blocks = &blocks[shared_prefix_len..];
            Ok(shared_prefix_len + self.insert_(child_id, key, blocks)?)
        } else {
            self.add_node(node_id, tokens, blocks);
            Ok(0)
        }
    }

    fn split_node(&mut self, node_id: NodeId, prefix_len: usize) -> NodeId {
        // We have to make the current node a child to ensure that its
        // properties and node id stay the same.

        // This funcion unwraps, an  invalid node_id is a programming error.

        let node = self
            .nodes
            .get_mut(node_id)
            .expect("Node to-be split does not exist");
        let mut parent_key = node.key.split_off(prefix_len);
        let mut parent_blocks = node.blocks.split_off(prefix_len);

        // Move first part of the prefix to the parent. We swap to avoid
        // an allocation + copy for both splits of the key/blocks.
        std::mem::swap(&mut node.key, &mut parent_key);
        std::mem::swap(&mut node.blocks, &mut parent_blocks);

        let node_key = node.key[0];

        let grandparent_id = node.parent.expect("Node does not have a parent");
        let parent_id = self.add_node(grandparent_id, parent_key, parent_blocks);
        self.add_node_to_parent(parent_id, node_key, node_id);

        // Reborrow to make the borrow checker happy.
        let node = self
            .nodes
            .get_mut(node_id)
            .expect("Node to-be split does not exist");
        node.parent = Some(parent_id);

        parent_id
    }

    /// Create a node and add it to the parent.
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

    /// Add a node to the parent.
    fn add_node_to_parent(&mut self, parent_id: NodeId, first: u32, child_id: NodeId) {
        // Unwrap here, passing in an unknown id is a programming error.
        let parent = self.nodes.get_mut(parent_id).expect("Unknown parent node");
        if parent.children.insert(first, child_id).is_none() {
            // Only increase reference count if child does not replace another child.
            self.incref(parent_id)
                .expect("Failed to increase parent refcount");
        }
    }

    /// Remove a node from the trie.
    fn remove_node(&mut self, node_id: NodeId) -> TrieNode {
        // Unwrap here, passing in an unknown id is a programming error.
        let node = self.nodes.remove(node_id).expect("Unknown node");
        let parent_id = node.parent.expect("Attempted to remove root node");
        let parent = self.nodes.get_mut(parent_id).expect("Unknown parent node");
        parent.children.remove(&node.key[0]);
        self.decref(parent_id)
            .expect("Failed to decrease parent refcount");
        self.nodes.remove(node_id);
        node
    }

    fn update_access_time(&mut self, node_id: NodeId) {
        // Unwrap here, passing in an unknown id is a programming error.
        let node = self.nodes.get_mut(node_id).expect("Unknown node");

        // Update the ordered leaves set if the node is a leave.
        if self.leaves.remove(&(node.last_accessed, node_id)) {
            self.leaves.insert((self.time, node_id));
        }

        node.last_accessed = self.time;
    }

    #[allow(dead_code)]
    #[doc(hidden)]
    /// Print debugging output for the trie.
    ///
    /// In contrast to `Debug` nicely formatted.
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

    pub(crate) fn root_id(&self) -> DefaultKey {
        self.root
    }
}

/// Trie node.
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

/// Helper trait to get the length of the shared prefix of two sequences.
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
    use std::sync::Arc;

    use crate::block_allocator::Allocator;

    use super::RadixAllocator;

    #[test]
    fn allocator_reuses_prefixes() {
        let mut cache = RadixAllocator::new(1, 12, None);
        let allocation = cache.allocate(8, Some(Arc::new(vec![0, 1, 2, 3]))).unwrap();
        assert_eq!(allocation.blocks, vec![4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(allocation.slots, allocation.slots);
        assert_eq!(allocation.prefix_len, 0);
        cache.free(allocation.blocks.clone(), allocation.allocation_id);

        let allocation = cache.allocate(8, Some(Arc::new(vec![0, 1, 2, 3]))).unwrap();
        assert_eq!(allocation.blocks, vec![4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(allocation.prefix_len, 4);
    }

    #[test]
    fn allocator_collects_older_prefixes_first() {
        let mut cache = RadixAllocator::new(1, 7, None);
        let allocation1 = cache.allocate(4, Some(Arc::new(vec![0, 1, 2, 3]))).unwrap();
        assert_eq!(allocation1.blocks, vec![3, 4, 5, 6]);
        assert_eq!(allocation1.prefix_len, 0);

        let allocation2 = cache.allocate(2, Some(Arc::new(vec![4, 5]))).unwrap();
        assert_eq!(allocation2.blocks, vec![1, 2]);
        assert_eq!(allocation2.prefix_len, 0);

        cache.free(allocation1.blocks.clone(), allocation1.allocation_id);
        cache.free(allocation2.blocks.clone(), allocation2.allocation_id);

        // We should get the blocks of the first allocation, since they are more recent.
        let allocation3 = cache.allocate(4, Some(Arc::new(vec![6, 7, 8, 9]))).unwrap();
        assert_eq!(allocation3.blocks, vec![3, 4, 5, 6]);
        assert_eq!(allocation3.prefix_len, 0);
    }

    #[test]
    fn allocator_frees_fully_overlapping_prefills() {
        let mut cache = RadixAllocator::new(1, 10, None);
        let allocation1 = cache.allocate(4, Some(Arc::new(vec![0, 1, 2, 3]))).unwrap();
        let allocation2 = cache.allocate(4, Some(Arc::new(vec![0, 1, 2, 3]))).unwrap();

        cache.free(allocation2.blocks.clone(), allocation2.allocation_id);
        cache.free(allocation1.blocks.clone(), allocation1.allocation_id);

        let allocation3 = cache.allocate(4, Some(Arc::new(vec![0, 1, 2, 3]))).unwrap();
        assert_eq!(allocation3.prefix_len, 4);

        // 10 blocks, of which 1 reserved for health checks, 4 for the cached blocks.
        assert_eq!(cache.free_blocks.len(), 5);
    }

    #[test]
    fn allocator_frees_partially_overlapping_prefills() {
        let mut cache = RadixAllocator::new(1, 20, None);
        let allocation1 = cache.allocate(4, Some(Arc::new(vec![0, 1]))).unwrap();
        assert_eq!(allocation1.blocks, vec![16, 17, 18, 19]);
        assert_eq!(allocation1.prefix_len, 0);

        cache.free(allocation1.blocks.clone(), allocation1.allocation_id);

        let allocation2 = cache
            .allocate(8, Some(Arc::new(vec![0, 1, 2, 3, 4, 5])))
            .unwrap();
        assert_eq!(allocation2.blocks, vec![16, 17, 12, 13, 14, 15, 18, 19]);
        assert_eq!(allocation2.prefix_len, 2);

        let allocation3 = cache
            .allocate(8, Some(Arc::new(vec![0, 1, 2, 3, 6, 7])))
            .unwrap();
        assert_eq!(allocation3.blocks, vec![16, 17, 6, 7, 8, 9, 10, 11]);
        assert_eq!(allocation3.prefix_len, 2);

        cache.free(allocation3.blocks.clone(), allocation3.allocation_id);
        cache.free(allocation2.blocks.clone(), allocation2.allocation_id);

        // 20 blocks, of which 1 reserved for health checks, 6 for allocation3, 2 for allocation2.
        assert_eq!(cache.free_blocks.len(), 11);

        let allocation4 = cache
            .allocate(6, Some(Arc::new(vec![0, 1, 2, 3, 4, 5])))
            .unwrap();
        assert_eq!(allocation4.blocks, vec![16, 17, 6, 7, 14, 15]);
        assert_eq!(allocation4.prefix_len, 6);
        assert_eq!(cache.free_blocks.len(), 11);

        let allocation5 = cache
            .allocate(6, Some(Arc::new(vec![0, 1, 2, 3, 6, 7])))
            .unwrap();
        assert_eq!(allocation5.blocks, vec![16, 17, 6, 7, 8, 9]);
        assert_eq!(allocation5.prefix_len, 6);
        assert_eq!(cache.free_blocks.len(), 11);
    }

    #[test]
    fn trie_insertions_have_correct_prefix_len() {
        let mut trie = super::RadixTrie::new();

        assert_eq!(trie.insert(&[0, 1, 2], &[0, 1, 2]).unwrap(), 0);

        // Already exists.
        assert_eq!(trie.insert(&[0, 1, 2], &[0, 1, 2]).unwrap(), 3);

        // Completely new at root-level
        assert_eq!(trie.insert(&[1, 2, 3], &[1, 2, 3]).unwrap(), 0);

        // Contains full prefix, but longer.
        assert_eq!(trie.insert(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap(), 3);

        // Shares partial prefix, we need a split.
        assert_eq!(
            trie.insert(&[0, 1, 2, 3, 5, 6, 7], &[0, 1, 2, 3, 5, 6, 7])
                .unwrap(),
            4
        );
    }

    #[test]
    fn trie_get_returns_correct_blocks() {
        let mut trie = super::RadixTrie::new();
        trie.insert(&[0, 1, 2], &[0, 1, 2]).unwrap();
        trie.insert(&[1, 2, 3], &[1, 2, 3]).unwrap();
        trie.insert(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
        trie.insert(&[0, 1, 2, 3, 5, 6, 7], &[0, 1, 2, 3, 5, 6, 7])
            .unwrap();

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
    fn trie_evict_removes_correct_blocks() {
        let mut trie = super::RadixTrie::new();
        trie.insert(&[0, 1, 2], &[0, 1, 2]).unwrap();
        trie.insert(&[0, 1, 2, 3, 5, 6, 7], &[0, 1, 2, 3, 5, 6, 7])
            .unwrap();
        trie.insert(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
        trie.insert(&[1, 2, 3], &[1, 2, 3]).unwrap();

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
