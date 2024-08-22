use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use crate::radix::RadixAllocator;

#[derive(Debug, Clone)]
pub struct BlockAllocation {
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
pub struct BlockAllocator {
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
    let mut allocator = RadixAllocator::new(block_size, blocks, window_size, prefix_caching);
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

// pub trait Allocator {
//     fn allocate(
//         &mut self,
//         tokens: u32,
//         prefill_tokens: Option<Arc<Vec<u32>>>,
//     ) -> Option<BlockAllocation>;
//
//     fn free(&mut self, blocks: Vec<u32>, allocation_id: u64);
// }
