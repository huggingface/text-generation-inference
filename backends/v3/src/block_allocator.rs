use std::{cmp::min, sync::Arc};
use tokio::sync::{mpsc, oneshot};

use crate::radix::RadixAllocator;

#[derive(Debug, Clone)]
pub(crate) struct BlockAllocation {
    pub allocation_id: u64,
    pub blocks: Vec<u32>,
    pub slots: Vec<u32>,

    /// Prefix that was cached and for which the KV does not have to
    /// be recomputed.
    pub prefix_len: u32,

    pub(crate) block_allocator: Option<BlockAllocator>,
}

impl Drop for BlockAllocation {
    fn drop(&mut self) {
        if let Some(block_allocator) = self.block_allocator.as_mut() {
            block_allocator.free(self.blocks.clone(), self.allocation_id)
        }
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
        prefix_caching: bool,
        window_size: Option<u32>,
    ) -> Self {
        // Create channel
        let (sender, receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::spawn(block_allocator_task(
            max_batch_total_tokens / block_size,
            block_size,
            prefix_caching,
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

        response_receiver.await.unwrap().map(|mut allocation| {
            allocation.block_allocator = Some(self.clone());
            allocation
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
    prefix_caching: bool,
    window_size: Option<u32>,
    mut receiver: mpsc::UnboundedReceiver<BlockAllocatorCommand>,
) {
    let mut allocator: Box<dyn Allocator + Send> = if prefix_caching {
        Box::new(RadixAllocator::new(block_size, blocks, window_size))
    } else {
        Box::new(SimpleAllocator::new(blocks, block_size, window_size))
    };
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
                response_sender
                    .send(allocator.allocate(tokens, prefill_tokens))
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
        response_sender: oneshot::Sender<Option<BlockAllocation>>,
    },
}

pub(crate) trait Allocator {
    fn allocate(
        &mut self,
        tokens: u32,
        prefill_tokens: Option<Arc<Vec<u32>>>,
    ) -> Option<BlockAllocation>;

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
        _prefill_tokens: Option<Arc<Vec<u32>>>,
    ) -> Option<BlockAllocation> {
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
            Some(BlockAllocation {
                allocation_id: 0,
                blocks,
                slots,
                prefix_len: 0,
                block_allocator: None,
            })
        }
    }

    fn free(&mut self, blocks: Vec<u32>, _allocation_id: u64) {
        self.free_blocks.extend(blocks)
    }
}
