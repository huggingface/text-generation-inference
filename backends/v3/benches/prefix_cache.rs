use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::Rng;

use text_generation_router_v3::block_allocator::{Allocator, RadixAllocator};

fn prefix_cache_benchmark(c: &mut Criterion) {
    let prefixes: Vec<Vec<u32>> = (0..8192)
        .chunks(256)
        .into_iter()
        .map(|c| c.collect())
        .collect();

    let mut cache = RadixAllocator::new(1, 262144, None);

    c.bench_function("fib 20", |b| {
        b.iter_batched(
            || {
                //prefixes
                //    .choose_multiple(&mut rand::thread_rng(), 5)
                //    .fold(Vec::new(), |mut v, s| {
                //        v.extend(s);
                //        v
                //    })

                (0..7936)
                    .map(|_| rand::thread_rng().gen_range(0..1024))
                    .collect::<Vec<u32>>()
            },
            |prefill| {
                let alloc = cache.allocate(prefill.len() as u32 + 13, Some(Arc::new(prefill)));
                if let Some(alloc) = alloc {
                    cache.free(alloc.0, alloc.3);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, prefix_cache_benchmark);
criterion_main!(benches);
