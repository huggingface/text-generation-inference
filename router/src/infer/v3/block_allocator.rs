use std::cmp::{max, min};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
pub(crate) struct BlockAllocation {
    pub blocks: Vec<u32>,
    pub slots: Vec<u32>,
    prompt_tokens: u32,
    decode_tokens: u32,
    block_allocator: BlockAllocator,
}

impl BlockAllocation {
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    pub(crate) async fn extend(&mut self, current_length: u32) -> Result<(), AllocationError> {
        let remaining_tokens = max(self.prompt_tokens + self.decode_tokens - current_length, 1);
        self.block_allocator
            .clone()
            .extend(self, remaining_tokens)
            .await
    }
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

    pub(crate) async fn allocate(
        &self,
        prompt_tokens: u32,
        decode_tokens: u32,
    ) -> Result<BlockAllocation, AllocationError> {
        let (response_sender, response_receiver) = oneshot::channel();
        self.block_allocator
            .send(BlockAllocatorCommand::Allocate {
                prompt_tokens,
                decode_tokens,
                response_sender,
            })
            .unwrap();

        response_receiver
            .await
            .unwrap()
            .map(|(blocks, slots)| BlockAllocation {
                blocks,
                slots,
                prompt_tokens,
                decode_tokens,
                block_allocator: self.clone(),
            })
    }

    pub(crate) async fn extend(
        &self,
        block_allocation: &mut BlockAllocation,
        tokens: u32,
    ) -> Result<(), AllocationError> {
        let (response_sender, response_receiver) = oneshot::channel();
        self.block_allocator
            .send(BlockAllocatorCommand::Allocate {
                prompt_tokens: 0,
                decode_tokens: tokens,
                response_sender,
            })
            .unwrap();

        let (blocks, slots) = response_receiver.await.unwrap()?;
        block_allocation.blocks.extend(blocks);
        block_allocation.slots.extend(slots);
        Ok(())
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
                prompt_tokens,
                decode_tokens,
                response_sender,
            } => {
                let decode_tokens = min(decode_tokens, block_size);
                let tokens = prompt_tokens + decode_tokens;

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

                let allocation = if required_blocks > free_blocks.len() as u32 {
                    Err(AllocationError::NotEnoughPages)
                } else {
                    let blocks =
                        free_blocks.split_off(free_blocks.len() - required_blocks as usize);
                    let mut slots = Vec::with_capacity(
                        (required_blocks * block_size * repeats as u32) as usize,
                    );

                    for block_id in blocks.repeat(repeats).iter() {
                        for s in (block_id * block_size)..((block_id + 1) * block_size) {
                            slots.push(s);
                        }
                    }
                    Ok((blocks, slots))
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
        prompt_tokens: u32,
        decode_tokens: u32,
        #[allow(clippy::type_complexity)]
        response_sender: oneshot::Sender<Result<(Vec<u32>, Vec<u32>), AllocationError>>,
    },
}

#[derive(Error, Debug)]
pub enum AllocationError {
    #[error("Not enough pages")]
    NotEnoughPages,
}
