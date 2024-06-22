use std::cmp::min;
use std::fmt::Formatter;
use std::sync::{Arc, Mutex, TryLockError};
use thiserror::Error;

#[derive(Clone)]
pub(crate) struct BlockAllocation {
    block_size: usize,
    allocated_blocks: Vec<u32>,
    required_blocks: usize,
    required_slots: usize,
    block_allocator: BlockAllocator,
}

impl BlockAllocation {
    pub(crate) fn len(&self) -> usize {
        self.allocated_blocks.len() * self.block_size
    }

    pub(crate) fn blocks(&self) -> &[u32] {
        &self.allocated_blocks
    }

    /// Extend an allocation by adding new blocks
    /// If the allocation length > window size, repeats blocks and slots to cover the
    /// whole `required_blocks` and `required_slots`
    pub(crate) fn extend(&mut self) -> Result<(), AllocationError> {
        let required_blocks = match self.block_allocator.window_size {
            None => self.required_blocks,
            Some(window_size) => min(
                (window_size as usize + self.block_size - 1) / self.block_size,
                self.required_blocks,
            ),
        };
        let remaining_blocks = required_blocks.saturating_sub(self.allocated_blocks.len());
        let new_blocks = min(remaining_blocks, 16);

        // Try to allocate all remaining blocks
        let blocks = match self.block_allocator.allocate_blocks(new_blocks) {
            Ok(blocks) => blocks,
            // Failed, try to allocate one block
            Err(_) => self.block_allocator.allocate_blocks(1)?,
        };
        // Add block and slots to current allocation
        self.allocated_blocks.extend(blocks);

        if let Some(window_size) = self.block_allocator.window_size {
            // if we have more slots than the window size,
            // we will never need to re-allocate and we can just repeat the blocks/slots
            let window_size = window_size as usize;
            if self.len() > window_size {
                let repeats = (self.required_slots + window_size - 1) / window_size;
                self.allocated_blocks = self.allocated_blocks.repeat(repeats);
                self.allocated_blocks.truncate(self.required_blocks);
            }
        }

        Ok(())
    }
}

impl Drop for BlockAllocation {
    /// Free the blocks
    fn drop(&mut self) {
        let allocated_blocks = std::mem::take(&mut self.allocated_blocks);
        self.block_allocator.free(allocated_blocks)
    }
}

impl std::fmt::Debug for BlockAllocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockAllocation")
            .field("allocated_blocks", &self.allocated_blocks.len())
            .field("required_blocks", &self.required_blocks)
            .field("required_slots", &self.required_slots)
            .field("block_allocator", &self.block_allocator)
            .finish()
    }
}

#[derive(Clone)]
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

    fn allocate_blocks(&self, blocks: usize) -> Result<Vec<u32>, AllocationError> {
        let mut free_blocks = self.free_blocks.lock().expect("Lock could not be acquired");

        if blocks > free_blocks.len() {
            // Not enough blocks to cover this allocation
            // Early return
            return Err(AllocationError::NotEnoughPages);
        }

        // Take the blocks
        let n_free_blocks = free_blocks.len();
        Ok(free_blocks.split_off(n_free_blocks - blocks))
    }

    /// For prompt tokens, we allocate enough blocks to cover all tokens
    /// For decode tokens, we allocate min(decode_blocks, 16) blocks
    ///
    /// If allocation > window size, we repeat blocks and slots
    pub(crate) fn block_allocation(
        &self,
        prompt_tokens: u32,
        decode_tokens: u32,
    ) -> Result<BlockAllocation, AllocationError> {
        let required_prompt_blocks = (prompt_tokens + self.block_size - 1) / self.block_size;
        // prompt blocks + 16 blocks for decode
        let decode_blocks = (decode_tokens + self.block_size - 1) / self.block_size;
        let required_blocks = required_prompt_blocks + min(decode_blocks, 16);
        let required_slots = required_blocks * self.block_size;

        // Slots and blocks required for the whole request
        let total_slots = prompt_tokens + decode_tokens;
        let total_required_blocks = (total_slots + self.block_size - 1) / self.block_size;

        let (clipped_required_blocks, repeats) = match self.window_size {
            Some(window_size) if required_slots >= window_size => {
                // Number of blocks for this window size
                let window_size_blocks = (window_size + self.block_size - 1) / self.block_size;
                // Number of times we will need to repeat blocks to cover the total allocation
                let repeats = (total_slots + window_size - 1) / window_size;
                (window_size_blocks, repeats)
            }
            // Nothing to do
            _ => (required_blocks, 1),
        };

        // Scoped to drop the lock early
        let allocated_blocks = {
            let mut free_blocks = self.free_blocks.lock().expect("Lock could not be acquired");
            let clipped_required_blocks = clipped_required_blocks as usize;

            if clipped_required_blocks > free_blocks.len() {
                // Not enough blocks to cover this allocation
                // Early return
                return Err(AllocationError::NotEnoughPages);
            }

            // Take the blocks
            let n_free_blocks = free_blocks.len();
            free_blocks.split_off(n_free_blocks - clipped_required_blocks)
        };

        let repeats = repeats as usize;
        let total_slots = total_slots as usize;
        let total_required_blocks = total_required_blocks as usize;

        let allocated_blocks = if repeats != 1 {
            let mut allocated_blocks = allocated_blocks.repeat(repeats);
            allocated_blocks.truncate(total_required_blocks);
            allocated_blocks
        } else {
            allocated_blocks
        };

        Ok(BlockAllocation {
            block_size: self.block_size as usize,
            allocated_blocks,
            required_blocks: total_required_blocks,
            required_slots: total_slots,
            block_allocator: self.clone(),
        })
    }

    pub(crate) fn free(&self, blocks: Vec<u32>) {
        self.free_blocks
            .lock()
            .expect("Lock could not be acquired. This is a bug.")
            .extend(blocks)
    }
}

impl std::fmt::Debug for BlockAllocator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("BlockAllocator");
        d.field("block_size", &self.block_size)
            .field("window_size", &self.window_size);
        match self.free_blocks.try_lock() {
            Ok(guard) => {
                d.field("free_blocks", &(*guard).len());
            }
            Err(TryLockError::Poisoned(err)) => {
                d.field("free_blocks", &(**err.get_ref()).len());
            }
            Err(TryLockError::WouldBlock) => {
                d.field("free_blocks", &format_args!("<locked>"));
            }
        };
        d.finish()
    }
}

#[derive(Error, Debug)]
pub enum AllocationError {
    #[error("Not enough pages")]
    NotEnoughPages,
}
