/// MIT License
//
// Copyright (c) 2020 hatoo
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
use std::collections::BTreeMap;

pub(crate) fn histogram(values: &[f64], bins: usize) -> Vec<(f64, usize)> {
    assert!(bins >= 2);
    let mut bucket: Vec<usize> = vec![0; bins];
    let min = values.iter().collect::<average::Min>().min();
    let max = values.iter().collect::<average::Max>().max();
    let step = (max - min) / (bins - 1) as f64;

    for &v in values {
        let i = std::cmp::min(((v - min) / step).ceil() as usize, bins - 1);
        bucket[i] += 1;
    }

    bucket
        .into_iter()
        .enumerate()
        .map(|(i, v)| (min + step * i as f64, v))
        .collect()
}

pub(crate) fn percentiles(values: &[f64], pecents: &[i32]) -> BTreeMap<String, f64> {
    pecents
        .iter()
        .map(|&p| {
            let i = (f64::from(p) / 100.0 * values.len() as f64) as usize;
            (format!("p{p}"), *values.get(i).unwrap_or(&f64::NAN))
        })
        .collect()
}
