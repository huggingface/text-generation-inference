use radix_trie::Trie;
use std::{
    cmp::min,
    collections::{hash_map::Entry, BTreeSet, HashMap},
    sync::Arc,
};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
pub(crate) struct BlockAllocation {
    pub blocks: Vec<u32>,
    pub slots: Vec<u32>,
    pub allocation_id: u64,
    block_allocator: BlockAllocator,
}

impl Drop for BlockAllocation {
    fn drop(&mut self) {
        self.block_allocator
            .free(self.blocks.clone(), self.allocation_id)
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

    pub(crate) async fn allocate(
        &self,
        tokens: u32,
        prefill_tokens: Option<Arc<Vec<u32>>>,
    ) -> Option<BlockAllocation> {
        let (response_sender, response_receiver) = oneshot::channel();
        self.block_allocator
            .send(BlockAllocatorCommand::Allocate {
                tokens,
                prefill_tokens,
                response_sender,
            })
            .unwrap();

        response_receiver
            .await
            .unwrap()
            .map(|(blocks, slots, allocation_id)| BlockAllocation {
                blocks,
                slots,
                allocation_id,
                block_allocator: self.clone(),
            })
    }

    pub(crate) fn free(&self, blocks: Vec<u32>, allocation_id: u64) {
        self.block_allocator
            .send(BlockAllocatorCommand::Free {
                allocation_id,
                blocks,
            })
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
            BlockAllocatorCommand::Free {
                blocks,
                allocation_id,
            } => allocator.free(blocks, allocation_id),
            BlockAllocatorCommand::Allocate {
                tokens,
                prefill_tokens,
                response_sender,
            } => {
                let prefill_tokens_slice = prefill_tokens.as_ref().map(|p| p.as_slice());
                response_sender
                    .send(allocator.allocate(tokens, prefill_tokens_slice))
                    .unwrap();
            }
        }
    }
}

#[derive(Debug)]
enum BlockAllocatorCommand {
    Free {
        blocks: Vec<u32>,
        allocation_id: u64,
    },
    Allocate {
        tokens: u32,
        prefill_tokens: Option<Arc<Vec<u32>>>,
        response_sender: oneshot::Sender<Option<(Vec<u32>, Vec<u32>, u64)>>,
    },
}

pub trait Allocator {
    fn allocate(
        &mut self,
        tokens: u32,
        prefill_tokens: Option<&[u32]>,
    ) -> Option<(Vec<u32>, Vec<u32>, u64)>;

    fn free(&mut self, blocks: Vec<u32>, allocation_id: u64);
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
    fn allocate(
        &mut self,
        tokens: u32,
        _prefill_tokens: Option<&[u32]>,
    ) -> Option<(Vec<u32>, Vec<u32>, u64)> {
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
            Some((blocks, slots, 0))
        }
    }

    fn free(&mut self, blocks: Vec<u32>, _allocation_id: u64) {
        self.free_blocks.extend(blocks)
    }
}

#[derive(Debug)]
struct PrefixBlockState {
    /// The block associated wit this prefix.
    block_id: u32,

    /// Last prefix block use.
    last_accessed: u64,

    ref_count: usize,
}

struct RadixAllocator {
    cache_blocks: Trie<Vec<u32>, ()>,

    /// Blocks that are immediately available for allocation.
    free_blocks: Vec<u32>,

    /// Prefix blocks with a reference count of zero, by staleness.
    leaves: BTreeSet<(u64, u64)>,

    // Avoid a system call, use a counter for time.
    time: u64,
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
            cache_blocks: Trie::new(),
            free_blocks: (1..n_blocks).collect(),
            leaves: BTreeSet::new(),
            time: 0,
        }
    }
}

#[derive(Debug)]
struct TrieNode {
    children: HashMap<u32, TrieNode>,
    key: Vec<u32>,
    blocks: Vec<u32>,
    last_accessed: u64,
}

impl TrieNode {
    fn new(key: Vec<u32>, blocks: Vec<u32>, last_accessed: u64) -> Self {
        TrieNode {
            children: HashMap::new(),
            key,
            blocks,
            last_accessed,
        }
    }

    // Insert a prefix into the trie. Returns the length of the shared prefix.
    fn insert(&mut self, key: &[u32], blocks: &[u32]) -> usize {
        match self.children.entry(key[0]) {
            Entry::Occupied(entry) => {
                let child = entry.into_mut();
                let shared_prefix_len = child
                    .key
                    .iter()
                    .zip(key)
                    .take_while(|(a, b)| a == b)
                    .count();

                // We are done, the prefix is already in the trie.
                if shared_prefix_len == key.len() {
                    return shared_prefix_len;
                }

                return shared_prefix_len
                    + child.insert(&key[shared_prefix_len..], &blocks[shared_prefix_len..]);
            }
            Entry::Vacant(_) => todo!(),
        }

        //node.last_accessed = last_accessed;
    }
}
