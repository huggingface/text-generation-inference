use itertools::Itertools;
use std::{
    borrow::BorrowMut,
    cmp::min,
    collections::{hash_map::Entry, HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
pub(crate) struct BlockAllocation {
    pub blocks: Vec<u32>,
    pub slots: Vec<u32>,
    block_allocator: BlockAllocator,
}

impl Drop for BlockAllocation {
    fn drop(&mut self) {
        self.block_allocator.free(self.blocks.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BlockAllocator {
    /// Channel to communicate with the background task
    block_allocator: mpsc::UnboundedSender<BlockAllocatorCommand>,
}

impl BlockAllocator {
    pub(crate) fn new(
        max_batch_total_tokens: u32,
        block_size: u32,
        window_size: Option<u32>,
    ) -> Self {
        // Create channel
        let (sender, receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::spawn(block_allocator_task(
            max_batch_total_tokens / block_size,
            block_size,
            window_size,
            receiver,
        ));

        Self {
            block_allocator: sender,
        }
    }

    pub(crate) async fn allocate(&self, tokens: u32) -> Option<BlockAllocation> {
        let (response_sender, response_receiver) = oneshot::channel();
        self.block_allocator
            .send(BlockAllocatorCommand::Allocate {
                tokens,
                response_sender,
            })
            .unwrap();

        response_receiver
            .await
            .unwrap()
            .map(|(blocks, slots)| BlockAllocation {
                blocks,
                slots,
                block_allocator: self.clone(),
            })
    }

    pub(crate) fn free(&self, blocks: Vec<u32>) {
        self.block_allocator
            .send(BlockAllocatorCommand::Free { blocks })
            .unwrap();
    }
}

async fn block_allocator_task(
    blocks: u32,
    block_size: u32,
    window_size: Option<u32>,
    mut receiver: mpsc::UnboundedReceiver<BlockAllocatorCommand>,
) {
    // Block 0 is reserved for health checks
    let mut free_blocks: Vec<u32> = (1..blocks).collect();
    while let Some(cmd) = receiver.recv().await {
        match cmd {
            BlockAllocatorCommand::Free { blocks } => free_blocks.extend(blocks),
            BlockAllocatorCommand::Allocate {
                tokens,
                response_sender,
            } => {
                // Apply window size
                let (required_blocks, repeats) = {
                    let (tokens, repeats) = match window_size {
                        None => (tokens, 1),
                        Some(window_size) => {
                            let repeats = (tokens + window_size - 1) / window_size;
                            let tokens = min(tokens, window_size);
                            (tokens, repeats as usize)
                        }
                    };
                    // Pad to a multiple of block size
                    let required_blocks = (tokens + block_size - 1) / block_size;
                    (required_blocks, repeats)
                };

                let tokens = tokens as usize;
                let allocation = if required_blocks > free_blocks.len() as u32 {
                    None
                } else {
                    let blocks =
                        free_blocks.split_off(free_blocks.len() - required_blocks as usize);
                    let mut slots = Vec::with_capacity(
                        (required_blocks * block_size * repeats as u32) as usize,
                    );

                    'slots: for block_id in blocks.repeat(repeats).iter() {
                        for s in (block_id * block_size)..((block_id + 1) * block_size) {
                            slots.push(s);
                            if slots.len() == tokens {
                                break 'slots;
                            }
                        }
                    }
                    Some((blocks, slots))
                };
                response_sender.send(allocation).unwrap();
            }
        }
    }
}

#[derive(Debug)]
enum BlockAllocatorCommand {
    Free {
        blocks: Vec<u32>,
    },
    Allocate {
        tokens: u32,
        response_sender: oneshot::Sender<Option<(Vec<u32>, Vec<u32>)>>,
    },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct BlockAllocationWithCache {
    pub blocks: Vec<u32>,
    pub slots: Vec<u32>,
}

#[derive(Debug)]
struct PrefixBlockState {
    block_id: u32,

    /// Last prefix block use.
    last_accessed: u64,

    /// Prefix predecessor (parent in the prefix trie).
    predecessor: Option<u64>,

    ref_count: usize,
}

#[derive(Debug)]
pub struct PrefixCache {
    /// Size of a paged attention block.
    block_size: usize,

    /// Blocks that cache a prefix with the given hash.
    ///
    /// The blocks form a Merkle tree, because a prefix block is dependent
    /// on its preceding prefix block.
    cache_blocks: HashMap<u64, PrefixBlockState>,

    /// Whether to cache partial blocks.
    cache_partial: bool,

    /// Blocks that are immediately available for allocation.
    free_blocks: Vec<u32>,

    /// Prefix blocks with a reference count of zero.
    leaves: HashSet<u64>,

    // Avoid a system call, use a counter for time.
    time: u64,
}

impl PrefixCache {
    pub fn new(block_size: usize, n_blocks: usize, cache_partial: bool) -> Self {
        PrefixCache {
            block_size,
            cache_blocks: HashMap::new(),
            cache_partial,
            free_blocks: (1..n_blocks as u32).collect(),
            leaves: HashSet::new(),
            time: 0,
        }
    }

    fn alloc(
        &mut self,
        n_tokens: usize,
        prefill_tokens: &[u32],
    ) -> Option<BlockAllocationWithCache> {
        let mut hasher = DefaultHasher::new();
        let mut tokens_from_cache = 0;
        let mut prefix_cache_blocks = Vec::new();
        let mut prefix_hashes = Vec::new();
        for prefill_chunk in prefill_tokens.chunks(self.block_size) {
            if prefill_chunk.len() < self.block_size && !self.cache_partial {
                break;
            }

            prefill_chunk.hash(&mut hasher);

            let prefix_hash = hasher.finish();

            let block_id = match self.cache_blocks.get(&prefix_hash) {
                Some(state) => state.block_id,
                None => break,
            };

            // We have to acquire the prefixes blocks, even if the allocation fails
            // later, otherwise the allocation below could garbage collect the
            // prefix blocks.
            self.incref_prefix(prefix_hash);
            prefix_hashes.push(prefix_hash);
            prefix_cache_blocks.push(block_id);

            tokens_from_cache += prefill_chunk.len();
        }

        // Get tokens for the remaining prefill and decode.
        let blocks = match self.alloc_or_reclaim(n_tokens - tokens_from_cache) {
            Some(blocks) => blocks,
            None => {
                // If the allocation fails, we have relinquish our use of the
                // prefix cache blocks. Maybe we can do this using `Drop`?
                for prefix_hash in prefix_hashes {
                    self.decref_prefix(prefix_hash);
                }

                return None;
            }
        };

        prefix_cache_blocks.extend(blocks);

        let mut slots = Vec::with_capacity(n_tokens);
        for block_id in prefix_cache_blocks.iter() {
            // TODO: fixme: doesn't work with cache_partial yet.
            for s in
                (*block_id * self.block_size as u32)..((*block_id + 1) * self.block_size as u32)
            {
                slots.push(s);
                if slots.len() == n_tokens {
                    break;
                }
            }
        }

        Some(BlockAllocationWithCache {
            blocks: prefix_cache_blocks,
            slots,
        })
    }

    fn free_prefix(&mut self, prefix_hash: u64) {
        let state = self
            .cache_blocks
            .remove(&prefix_hash)
            .expect("Unknown hash");

        // Parent has one user less.
        if let Some(predecessor) = state.predecessor {
            self.decref_prefix(predecessor);
        }

        self.leaves.remove(&prefix_hash);
    }

    fn decref_prefix(&mut self, prefix_hash: u64) {
        let state = self
            .cache_blocks
            .get_mut(&prefix_hash)
            .expect("Unknown hash");
        assert!(state.ref_count > 0);
        state.ref_count -= 1;
        if state.ref_count == 0 {
            self.leaves.insert(prefix_hash);
        }
    }

    fn incref_prefix(&mut self, prefix_hash: u64) {
        let state = self
            .cache_blocks
            .get_mut(&prefix_hash)
            .expect("Unknown hash");
        state.ref_count += 1;
        self.leaves.remove(&prefix_hash);
    }

    fn alloc_or_reclaim(&mut self, n_tokens: usize) -> Option<Vec<u32>> {
        let n_blocks = (n_tokens + self.block_size - 1) / self.block_size;
        let n_blocks_needed = if n_tokens > self.free_blocks.len() {
            n_blocks - self.free_blocks.len()
        } else {
            0
        };

        while self.free_blocks.len() < n_blocks_needed {
            // We have to free one block at a time, because removing the LRU
            // prefix block may make available another prefix block that is
            // LRU.
            //
            // TODO: switch to something like a binary heap to avoid sorting
            // the set of leaves over and over again.

            let (lru_prefix_hash, lru_block_id) = self
                .leaves
                .iter()
                .map(|prefix_hash| (prefix_hash, &self.cache_blocks[prefix_hash]))
                .sorted_by_key(|state| state.1.last_accessed)
                .map(|(prefix_hash, state)| (*prefix_hash, state.block_id))
                .next()?;

            self.free_prefix(lru_prefix_hash);
            self.free_blocks.push(lru_block_id);
        }

        Some(
            self.free_blocks
                .split_off(self.free_blocks.len() - n_blocks),
        )
    }

    fn free(&mut self, blocks: &[u32], prefill_tokens: &[u32]) {
        let mut hasher = DefaultHasher::new();
        let mut predecessor = None;
        for (prefill_chunk, block_id) in prefill_tokens.chunks(self.block_size).zip(blocks.iter()) {
            if prefill_chunk.len() < self.block_size && !self.cache_partial {
                break;
            }

            prefill_chunk.hash(&mut hasher);
            let prefix_hash = hasher.finish();

            match self.cache_blocks.entry(prefix_hash) {
                Entry::Occupied(mut entry) => {
                    let value = entry.get_mut();
                    value.last_accessed = self.time;
                    self.decref_prefix(prefix_hash);
                }
                Entry::Vacant(entry) => {
                    entry.insert(PrefixBlockState {
                        block_id: *block_id,
                        last_accessed: self.time,
                        predecessor,
                        ref_count: 0,
                    });
                    self.leaves.insert(prefix_hash);
                }
            };

            predecessor = Some(prefix_hash);
        }

        self.time += 1;

        let n_prefill_blocks = (prefill_tokens.len() + self.block_size - 1) / self.block_size;
        for block in &blocks[n_prefill_blocks..] {
            self.free_blocks.push(*block);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::infer::v3::block_allocator::BlockAllocationWithCache;

    use super::PrefixCache;

    #[test]
    fn test_prefix_cache() {
        let mut cache = PrefixCache::new(4, 3, false);
        let allocation = cache.alloc(8, &[0, 1, 2, 3]);
        assert_eq!(
            allocation,
            Some(BlockAllocationWithCache {
                blocks: vec![1, 2],
                slots: (4..12).collect()
            })
        );
        cache.free(&allocation.unwrap().blocks, &[0, 1, 2, 3]);

        eprintln!("{:?}", cache);

        let allocation = cache.alloc(8, &[0, 1, 2, 3]);
        assert_eq!(
            allocation,
            Some(BlockAllocationWithCache {
                blocks: vec![1, 2],
                slots: (4..12).collect()
            })
        );
    }
}
