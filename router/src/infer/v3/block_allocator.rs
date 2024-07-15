use std::{
    cmp::min,
    collections::{hash_map::Entry, BTreeSet, HashMap},
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
    let mut allocator = SimpleAllocator::new(blocks, block_size, window_size);
    while let Some(cmd) = receiver.recv().await {
        match cmd {
            BlockAllocatorCommand::Free { blocks } => allocator.free(blocks),
            BlockAllocatorCommand::Allocate {
                tokens,
                response_sender,
            } => {
                response_sender.send(allocator.allocate(tokens)).unwrap();
            }
        }
    }
}

pub trait Allocator {
    fn allocate(&mut self, tokens: u32) -> Option<(Vec<u32>, Vec<u32>)>;

    fn free(&mut self, blocks: Vec<u32>);
}

pub struct SimpleAllocator {
    free_blocks: Vec<u32>,
    block_size: u32,
    window_size: Option<u32>,
}

impl SimpleAllocator {
    fn new(blocks: u32, block_size: u32, window_size: Option<u32>) -> Self {
        SimpleAllocator {
            block_size,
            // Block 0 is reserved for health checks
            free_blocks: (1..blocks).collect(),
            window_size,
        }
    }
}

impl Allocator for SimpleAllocator {
    fn allocate(&mut self, tokens: u32) -> Option<(Vec<u32>, Vec<u32>)> {
        // Apply window size
        let (required_blocks, repeats) = {
            let (tokens, repeats) = match self.window_size {
                None => (tokens, 1),
                Some(window_size) => {
                    let repeats = (tokens + window_size - 1) / window_size;
                    let tokens = min(tokens, window_size);
                    (tokens, repeats as usize)
                }
            };
            // Pad to a multiple of block size
            let required_blocks = (tokens + self.block_size - 1) / self.block_size;
            (required_blocks, repeats)
        };

        let tokens = tokens as usize;
        if required_blocks > self.free_blocks.len() as u32 {
            None
        } else {
            let blocks = self
                .free_blocks
                .split_off(self.free_blocks.len() - required_blocks as usize);
            let mut slots =
                Vec::with_capacity((required_blocks * self.block_size * repeats as u32) as usize);

            'slots: for block_id in blocks.repeat(repeats).iter() {
                for s in (block_id * self.block_size)..((block_id + 1) * self.block_size) {
                    slots.push(s);
                    if slots.len() == tokens {
                        break 'slots;
                    }
                }
            }
            Some((blocks, slots))
        }
    }

    fn free(&mut self, blocks: Vec<u32>) {
        self.free_blocks.extend(blocks)
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
pub struct BlockAllocationWithCache {
    pub blocks: Vec<u32>,
    pub slots: Vec<u32>,
    pub prefix_hashes: Vec<u64>,
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
pub struct PrefixCacheAllocator {
    /// Size of a paged attention block.
    block_size: usize,

    /// Blocks that cache a prefix with the given hash.
    ///
    /// The blocks form a Merkle tree, because a prefix block is dependent
    /// on its preceding prefix block.
    cache_blocks: HashMap<u64, PrefixBlockState>,

    /// Blocks that are immediately available for allocation.
    free_blocks: Vec<u32>,

    /// Prefix blocks with a reference count of zero, by staleness.
    leaves: BTreeSet<(u64, u64)>,

    // Avoid a system call, use a counter for time.
    time: u64,
}

impl PrefixCacheAllocator {
    pub fn new(block_size: usize, n_blocks: usize, window_size: Option<u32>) -> Self {
        if window_size.is_some() {
            unimplemented!("Window size not supported in the prefix-caching block allocator yet");
        }

        PrefixCacheAllocator {
            block_size,
            cache_blocks: HashMap::new(),
            free_blocks: (1..n_blocks as u32).collect(),
            leaves: BTreeSet::new(),
            time: 0,
        }
    }

    pub fn alloc(
        &mut self,
        n_tokens: usize,
        prefill_tokens: &[u32],
    ) -> Option<BlockAllocationWithCache> {
        let mut hasher = DefaultHasher::new();
        let mut prefix_cache_blocks = Vec::new();
        let mut prefix_hashes = Vec::new();
        for prefill_chunk in prefill_tokens.chunks(self.block_size) {
            if prefill_chunk.len() < self.block_size {
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
        }

        // Get tokens for the remaining prefill and decode.
        let blocks = match self.alloc_or_reclaim(n_tokens - (prefix_hashes.len() * self.block_size))
        {
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
            prefix_hashes,
        })
    }

    fn free_prefix_block(&mut self, prefix_hash: u64) {
        let state = self
            .cache_blocks
            .remove(&prefix_hash)
            .expect("Unknown hash");

        // Parent has one user less.
        if let Some(predecessor) = state.predecessor {
            self.decref_prefix(predecessor);
        }

        self.leaves.remove(&(state.last_accessed, prefix_hash));
        self.free_blocks.push(state.block_id);
    }

    fn decref_prefix(&mut self, prefix_hash: u64) {
        let state = self
            .cache_blocks
            .get_mut(&prefix_hash)
            .expect("Unknown hash");
        assert!(state.ref_count > 0);
        state.ref_count -= 1;
        if state.ref_count == 0 {
            self.leaves.insert((state.last_accessed, prefix_hash));
        }
    }

    fn incref_prefix(&mut self, prefix_hash: u64) {
        let state = self
            .cache_blocks
            .get_mut(&prefix_hash)
            .expect("Unknown hash");
        state.ref_count += 1;
        self.leaves.remove(&(state.last_accessed, prefix_hash));
    }

    fn alloc_or_reclaim(&mut self, n_tokens: usize) -> Option<Vec<u32>> {
        let n_blocks_needed = (n_tokens + self.block_size - 1) / self.block_size;

        while self.free_blocks.len() < n_blocks_needed {
            // We have to free one block at a time because removing the LRU
            // prefix block may make available another prefix block that is
            // LRU.
            let (_, lru_prefix_hash) = self.leaves.pop_first()?;
            self.free_prefix_block(lru_prefix_hash);
        }

        Some(
            self.free_blocks
                .split_off(self.free_blocks.len() - n_blocks_needed),
        )
    }

    pub fn free(&mut self, blocks: &[u32], prefix_hashes: &[u64]) {
        let mut predecessor = None;

        for (&prefix_hash, &block_id) in prefix_hashes.iter().zip(blocks.iter()) {
            match self.cache_blocks.entry(prefix_hash) {
                Entry::Occupied(mut entry) => {
                    let value = entry.get_mut();
                    value.last_accessed = self.time;
                    self.decref_prefix(prefix_hash);
                }
                Entry::Vacant(entry) => {
                    entry.insert(PrefixBlockState {
                        block_id,
                        last_accessed: self.time,
                        predecessor,
                        ref_count: 0,
                    });
                    if let Some(predecessor) = predecessor {
                        self.incref_prefix(predecessor);
                    }
                    self.leaves.insert((self.time, prefix_hash));
                }
            };

            predecessor = Some(prefix_hash);
        }

        self.time += 1;

        for block in &blocks[prefix_hashes.len()..] {
            self.free_blocks.push(*block);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::infer::v3::block_allocator::BlockAllocationWithCache;

    use super::PrefixCacheAllocator;

    #[test]
    fn test_prefix_cache() {
        let mut cache = PrefixCacheAllocator::new(4, 3, None);
        let allocation = cache.alloc(8, &[0, 1, 2, 3]);
        assert_eq!(
            allocation,
            Some(BlockAllocationWithCache {
                blocks: vec![1, 2],
                slots: (4..12).collect(),
                prefix_hashes: Vec::new(),
            })
        );
        cache.free(&allocation.unwrap().blocks, &[0, 1, 2, 3]);

        let allocation = cache.alloc(8, &[0, 1, 2, 3]);
        assert_eq!(
            allocation,
            Some(BlockAllocationWithCache {
                blocks: vec![1, 2],
                slots: (4..12).collect(),
                prefix_hashes: Vec::new(),
            })
        );
    }

    #[test]
    fn test_older_prefixes_are_collected_first() {
        let mut cache = PrefixCacheAllocator::new(2, 4, None);
        let allocation1 = cache.alloc(4, &[0, 1, 2, 3]);
        assert_eq!(
            allocation1,
            Some(BlockAllocationWithCache {
                blocks: vec![2, 3],
                slots: (4..8).collect(),
                prefix_hashes: Vec::new(),
            })
        );

        let allocation2 = cache.alloc(2, &[4, 5]);
        assert_eq!(
            allocation2,
            Some(BlockAllocationWithCache {
                blocks: vec![1],
                slots: (2..4).collect(),
                prefix_hashes: Vec::new(),
            })
        );

        cache.free(&allocation1.unwrap().blocks, &[0, 1, 2, 3]);
        cache.free(&allocation2.unwrap().blocks, &[4, 5]);

        // We should get the blocks of the first allocation, since they are more recent.
        let allocation3 = cache.alloc(4, &[6, 7, 8, 9]);
        assert_eq!(
            allocation3,
            Some(BlockAllocationWithCache {
                blocks: vec![3, 2],
                slots: vec![6, 7, 4, 5],
                prefix_hashes: Vec::new(),
            })
        );
    }
}
