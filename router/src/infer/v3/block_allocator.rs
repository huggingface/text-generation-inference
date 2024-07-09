use itertools::Itertools;
use std::{
    borrow::BorrowMut,
    cmp::min,
    collections::{hash_map::Entry, HashMap},
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
struct BlockState {
    block_id: u32,
    last_accessed: u64,
    ref_count: usize,
}

pub struct PrefixCache {
    block_size: usize,
    cache_partial: bool,
    free_blocks: Vec<u32>,
    cache_blocks: HashMap<u64, BlockState>,

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
            time: 0,
        }
    }

    fn alloc(
        &mut self,
        mut n_tokens: usize,
        prefill_tokens: &[u32],
    ) -> Option<BlockAllocationWithCache> {
        // First try to lookup prefix.
        let mut hasher = DefaultHasher::new();
        let mut tokens_from_cache = 0;
        let mut cache_blocks_hashes = Vec::new();
        for prefill_chunk in prefill_tokens.chunks(self.block_size) {
            if prefill_chunk.len() < self.block_size && !self.cache_partial {
                break;
            }

            prefill_chunk.hash(&mut hasher);

            match self.cache_blocks.get_mut(&hasher.finish()) {
                Some(state) => {
                    state.ref_count += 1;
                }
                None => todo!(),
            }

            if !self.cache_blocks.contains_key(&hasher.finish()) {
                break;
            }

            tokens_from_cache += prefill_chunk.len();
            cache_blocks_hashes.push(hasher.finish());
        }

        let blocks = self.alloc_or_reclaim(n_tokens - tokens_from_cache)?;

        let mut prefix_cache_blocks = Vec::new();
        for hash in cache_blocks_hashes {
            match self.cache_blocks.get_mut(&hash) {
                Some(info) => {
                    info.ref_count += 1;
                    prefix_cache_blocks.push(info.block_id);
                }
                None => unreachable!(),
            }
        }

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
        })
    }

    fn alloc_or_reclaim(&mut self, n_tokens: usize) -> Option<Vec<u32>> {
        let n_blocks = (n_tokens + self.block_size - 1) / self.block_size;
        let n_blocks_needed = if n_tokens > self.free_blocks.len() {
            n_blocks - self.free_blocks.len()
        } else {
            0
        };

        if n_blocks_needed > 0 {
            let removable_blocks = self
                .cache_blocks
                .iter_mut()
                // Block must be unused.
                .filter(|(_, state)| state.ref_count == 0)
                // Remove most recent block first.
                // TODO: we are not yet removing a prefix in reverse order.
                .sorted_by_key(|(_, state)| state.last_accessed)
                // Find enough candidates.
                .take(n_blocks_needed)
                .map(|(block_hash, block_state)| (*block_hash, block_state.block_id))
                .collect::<Vec<_>>();

            if removable_blocks.len() < n_blocks_needed {
                return None;
            }

            for (block_hash, block_id) in removable_blocks.into_iter() {
                self.free_blocks.push(block_id);
                self.cache_blocks.remove(&block_hash);
            }
        }

        Some(
            self.free_blocks
                .split_off(self.free_blocks.len() - n_blocks),
        )
    }

    fn free(&mut self, blocks: &[u32], prefill_tokens: &[u32]) {
        let mut hasher = DefaultHasher::new();

        for (prefill_chunk, block_id) in prefill_tokens.chunks(self.block_size).zip(blocks.iter()) {
            if prefill_chunk.len() < self.block_size && !self.cache_partial {
                break;
            }

            prefill_chunk.hash(&mut hasher);

            match self.cache_blocks.entry(hasher.finish()) {
                Entry::Occupied(mut entry) => {
                    let value = entry.get_mut();
                    value.last_accessed = self.time;
                    assert!(value.ref_count > 0);
                    value.ref_count -= 1;
                }
                Entry::Vacant(entry) => {
                    entry.insert(BlockState {
                        block_id: *block_id,
                        last_accessed: self.time,
                        ref_count: 0,
                    });
                }
            };
        }

        self.time += 1;
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
