use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Clone)]
pub(crate) struct BlockAllocation {
    allocated_blocks: Vec<u32>,
    allocated_slots: Vec<u32>,
    required_blocks: usize,
    required_slots: usize,
    block_allocator: BlockAllocator,
}

impl BlockAllocation {
    pub(crate) fn len(&self) -> usize {
        self.allocated_slots.len()
    }

    pub(crate) fn blocks(&self) -> &[u32] {
        &self.allocated_blocks
    }

    pub(crate) fn slots(&self) -> &[u32] {
        &self.allocated_slots
    }

    pub(crate) fn extend(&mut self) -> Result<(), AllocationError> {
        let (block, slots) = self.block_allocator.allocate_block()?;
        // Add block and slots to current allocation
        self.allocated_blocks.push(block);
        self.allocated_slots.extend(slots);

        if let Some(window_size) = self.block_allocator.window_size {
            // if we have more slots than the window size,
            // we will never need to re-allocate and we can just repeat the blocks/slots
            let window_size = window_size as usize;
            if self.len() > window_size {
                let repeats = (self.required_slots + window_size - 1) / window_size;
                self.allocated_blocks = self.allocated_blocks.repeat(repeats);
                self.allocated_blocks.truncate(self.required_blocks);
                self.allocated_slots = self.allocated_slots.repeat(repeats);
                self.allocated_slots.truncate(self.required_slots);
            }
        }

        Ok(())
    }
}

impl Drop for BlockAllocation {
    fn drop(&mut self) {
        let allocated_blocks = std::mem::take(&mut self.allocated_blocks);
        self.block_allocator.free(allocated_blocks)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BlockAllocator {
    free_blocks: Arc<Mutex<Vec<u32>>>,
    block_size: u32,
    window_size: Option<u32>,
}

impl BlockAllocator {
    pub(crate) fn new(
        max_batch_total_tokens: u32,
        block_size: u32,
        window_size: Option<u32>,
    ) -> Self {
        let blocks = max_batch_total_tokens / block_size;
        // Block 0 is reserved for health checks
        let free_blocks: Vec<u32> = (1..blocks).collect();

        Self {
            free_blocks: Arc::new(Mutex::new(free_blocks)),
            block_size,
            window_size,
        }
    }

    fn allocate_block(&self) -> Result<(u32, Vec<u32>), AllocationError> {
        let mut free_blocks = self.free_blocks.lock().expect("Lock could not be acquired");

        if free_blocks.is_empty() {
            return Err(AllocationError::NotEnoughPages);
        }

        let block_id = free_blocks.pop().unwrap();
        let slots = ((block_id * self.block_size)..((block_id + 1) * self.block_size)).collect();
        Ok((block_id, slots))
    }

    /// For prompt tokens, we allocate enough blocks to cover all tokens
    /// For decode tokens, we allocate block by block
    ///
    /// If prompt tokens + min(decode_tokens, block_size) > window size, we repeat blocks and slots
    pub(crate) fn block_allocation(
        &self,
        prompt_tokens: u32,
        decode_tokens: u32,
    ) -> Result<BlockAllocation, AllocationError> {
        let required_prompt_blocks = (prompt_tokens + self.block_size - 1) / self.block_size;
        // prompt blocks + a single block for decode
        let required_blocks = required_prompt_blocks + 1;

        let (clipped_required_blocks, repeats) = match self.window_size {
            // Nothing to do
            None => (required_blocks, 1),
            Some(window_size) => {
                // Number of blocks for this window size
                let window_size_blocks = (window_size + self.block_size - 1) / self.block_size;

                if required_blocks > window_size_blocks {
                    // Number of times we will need to repeat blocks to cover the required allocation
                    let repeats = (required_blocks + window_size_blocks - 1) / window_size_blocks;
                    (window_size_blocks, repeats)
                } else {
                    (required_blocks, 1)
                }
            }
        };

        let repeats = repeats as usize;
        let required_blocks = required_blocks as usize;
        let clipped_required_blocks = clipped_required_blocks as usize;

        let mut free_blocks = self.free_blocks.lock().expect("Lock could not be acquired");

        if clipped_required_blocks > free_blocks.len() {
            Err(AllocationError::NotEnoughPages)
        } else {
            let n_free_blocks = free_blocks.len();
            let allocated_blocks =
                free_blocks.split_off(n_free_blocks - clipped_required_blocks);

            let allocated_blocks = if repeats != 1 {
                let mut allocated_blocks = allocated_blocks.repeat(repeats);
                allocated_blocks.truncate(required_blocks);
                allocated_blocks
            } else {
                allocated_blocks
            };

            let mut allocated_slots = Vec::with_capacity(
                allocated_blocks.len() * self.block_size as usize * repeats,
            );

            let required_slots = (prompt_tokens + decode_tokens) as usize;

            'slots: for block_id in allocated_blocks.iter() {
                for s in (block_id * self.block_size)..((block_id + 1) * self.block_size) {
                    allocated_slots.push(s);
                    if allocated_slots.len() > required_slots {
                        break 'slots;
                    }
                }
            }

            Ok(BlockAllocation {
                allocated_blocks,
                allocated_slots,
                required_blocks,
                required_slots,
                block_allocator: self.clone(),
            })
        }
    }

    pub(crate) fn free(&self, blocks: Vec<u32>) {
        self.free_blocks.lock().expect("Lock could not be acquired. This is a bug.").extend(blocks)
    }
}

#[derive(Error, Debug)]
pub enum AllocationError {
    #[error("Not enough pages")]
    NotEnoughPages,
}
