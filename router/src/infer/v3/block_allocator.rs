use std::cmp::min;
use std::sync::{Arc, Mutex};
use thiserror::Error;

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

    pub(crate) fn extend(&mut self) -> Result<(), AllocationError> {
        let (block, slots) = self.block_allocator.allocate_block()?;

        match self.block_allocator.window_size {
            None => {
                self.blocks.push(block);
                self.slots.extend(slots);
            }
            Some(window_size) => {
                if self.len() as u32 > window_size {
                    let total_tokens = self.prompt_tokens + self.decode_tokens;

                    let repeats = (total_tokens + window_size - 1) / window_size;
                }
            }
        }
        Ok(())
    }
}

impl Drop for BlockAllocation {
    fn drop(&mut self) {
        self.block_allocator.free(self.blocks.clone())
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
    fn allocate(
        &self,
        prompt_tokens: u32,
        decode_tokens: u32,
    ) -> Result<(Vec<u32>, Vec<u32>), AllocationError> {
        // let decode_tokens = min(decode_tokens, self.block_size);
        // let tokens = prompt_tokens + decode_tokens;

        let required_prompt_blocks = (prompt_tokens + self.block_size - 1) / self.block_size;
        // prompt blocks + a single block for decode
        let required_blocks = required_prompt_blocks + 1;

        let (required_blocks, repeats) = match self.window_size {
            // Nothing to do
            None => (required_blocks, 1),
            Some(window_size) => {
                // Number of blocks needed for this window size
                let window_size_required_blocks = (window_size + self.block_size - 1) / self.block_size;
                // Number of times we will need to repeat blocks to cover the required allocation
                let repeats = (required_blocks + window_size_required_blocks -1) / window_size_required_blocks;
                let required_blocks = min(required_blocks, window_size_required_blocks);

                (required_blocks, repeats)
            }
        };


        /// if prompt + decode < window size => do nothing
        /// if prompt + decode > window size => do normal until we reach window size then

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

        let mut free_blocks = self.free_blocks.lock().expect("Lock could not be acquired");

        if required_blocks > free_blocks.len() as u32 {
            Err(AllocationError::NotEnoughPages)
        } else {
            let n_free_blocks = free_blocks.len();
            let blocks =
                free_blocks.split_off(n_free_blocks - required_blocks as usize);
            let mut slots = Vec::with_capacity(
                (required_blocks * self.block_size * repeats as u32) as usize,
            );

            for block_id in blocks.repeat(repeats).iter() {
                for s in (block_id * self.block_size)..((block_id + 1) * self.block_size) {
                    slots.push(s);
                }
            }
            Ok((blocks, slots))
        }
    }

    pub(crate) fn block_allocation(
        &self,
        prompt_tokens: u32,
        decode_tokens: u32,
    ) -> Result<BlockAllocation, AllocationError> {
        self.allocate_inner(prompt_tokens, decode_tokens)
            .map(|(blocks, slots)| BlockAllocation {
                blocks,
                slots,
                prompt_tokens,
                decode_tokens,
                block_allocator: self.clone(),
            })
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
