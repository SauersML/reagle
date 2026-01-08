use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use reagle::model::hmm::HmmUpdater;
use std::hint::black_box;

/// Benchmark HMM forward update with different state counts
fn bench_fwd_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmm_fwd_update");

    for n_states in [64, 256, 1024, 4096] {
        group.throughput(Throughput::Elements(n_states as u64));

        group.bench_with_input(
            BenchmarkId::new("states", n_states),
            &n_states,
            |b, &n_states| {
                let mut fwd = vec![1.0 / n_states as f32; n_states];
                let emit_probs = [0.99, 0.01];
                let mismatches: Vec<u8> = (0..n_states).map(|i| (i % 2) as u8).collect();
                let fwd_sum = 1.0f32;
                let p_switch = 0.01f32;

                b.iter(|| {
                    let sum = HmmUpdater::fwd_update(
                        black_box(&mut fwd),
                        black_box(fwd_sum),
                        black_box(p_switch),
                        black_box(&emit_probs),
                        black_box(&mismatches),
                        black_box(n_states),
                    );
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark HMM backward update with different state counts
fn bench_bwd_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmm_bwd_update");

    for n_states in [64, 256, 1024, 4096] {
        group.throughput(Throughput::Elements(n_states as u64));

        group.bench_with_input(
            BenchmarkId::new("states", n_states),
            &n_states,
            |b, &n_states| {
                let mut bwd = vec![1.0 / n_states as f32; n_states];
                let emit_probs = [0.99, 0.01];
                let mismatches: Vec<u8> = (0..n_states).map(|i| (i % 2) as u8).collect();
                let p_switch = 0.01f32;

                b.iter(|| {
                    HmmUpdater::bwd_update(
                        black_box(&mut bwd),
                        black_box(p_switch),
                        black_box(&emit_probs),
                        black_box(&mismatches),
                        black_box(n_states),
                    );
                })
            },
        );
    }

    group.finish();
}

/// Benchmark full forward-backward pass scaling
fn bench_forward_backward_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmm_forward_backward");
    group.sample_size(50);

    // Typical imputation: 1600 states, varying markers
    let n_states = 1600;

    for n_markers in [100, 500, 1000, 5000] {
        group.throughput(Throughput::Elements((n_markers * n_states) as u64));

        group.bench_with_input(
            BenchmarkId::new("markers", n_markers),
            &n_markers,
            |b, &n_markers| {
                let mut fwd = vec![vec![1.0 / n_states as f32; n_states]; n_markers];
                let mut bwd = vec![1.0 / n_states as f32; n_states];
                let emit_probs = [0.99, 0.01];
                let mismatches: Vec<Vec<u8>> = (0..n_markers)
                    .map(|_| (0..n_states).map(|i| (i % 2) as u8).collect())
                    .collect();
                let p_switch = 0.01f32;

                b.iter(|| {
                    // Forward pass
                    let mut fwd_sum = 1.0f32;
                    for m in 0..n_markers {
                        fwd_sum = HmmUpdater::fwd_update(
                            &mut fwd[m],
                            fwd_sum,
                            p_switch,
                            &emit_probs,
                            &mismatches[m],
                            n_states,
                        );
                    }

                    // Backward pass
                    for m in (0..n_markers).rev() {
                        HmmUpdater::bwd_update(
                            &mut bwd,
                            p_switch,
                            &emit_probs,
                            &mismatches[m],
                            n_states,
                        );
                    }

                    black_box(fwd_sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark ThreadedHaps segment lookup (cursor-style traversal)
fn bench_threaded_haps_traversal(c: &mut Criterion) {
    use reagle::ThreadedHaps;

    let mut group = c.benchmark_group("threaded_haps");

    let n_markers = 10000;
    let n_states = 1600;
    let segments_per_state = 10; // ~1000 markers per segment

    // Build threaded haps with realistic segment structure
    let mut th = ThreadedHaps::new(n_states, n_states * (segments_per_state + 1), n_markers);
    for state in 0..n_states {
        th.push_new(state as u32);
        // Add segment transitions at regular intervals
        for seg in 1..segments_per_state {
            let boundary = (n_markers / segments_per_state) * seg;
            th.add_segment(state, (state + seg) as u32, boundary);
        }
    }

    group.throughput(Throughput::Elements((n_markers * n_states) as u64));

    group.bench_function("full_traversal", |b| {
        b.iter(|| {
            th.reset_cursors();
            let mut sum = 0u32;
            for m in 0..n_markers {
                for state in 0..n_states {
                    sum = sum.wrapping_add(th.hap_at_raw(state, m));
                }
            }
            black_box(sum)
        })
    });

    group.finish();
}

/// Benchmark allele mismatch computation (emission probability setup)
fn bench_mismatch_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mismatch_computation");

    for n_states in [256, 1024, 4096] {
        group.throughput(Throughput::Elements(n_states as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", n_states),
            &n_states,
            |b, &n_states| {
                let ref_alleles: Vec<u8> = (0..n_states).map(|i| (i % 2) as u8).collect();
                let target_allele = 0u8;
                let mut mismatches = vec![0u8; n_states];

                b.iter(|| {
                    for k in 0..n_states {
                        mismatches[k] = if ref_alleles[k] == target_allele { 0 } else { 1 };
                    }
                    black_box(mismatches.iter().sum::<u8>())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory access patterns (sequential vs random)
fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access");
    group.sample_size(50);

    // Simulate state probability array access patterns
    let n_markers = 5000;
    let n_states = 1600;
    let data: Vec<f32> = (0..n_markers * n_states).map(|i| (i as f32) * 0.0001).collect();

    group.throughput(Throughput::Elements((n_markers * n_states) as u64));

    // Sequential access (marker-major: what cursor does)
    group.bench_function("sequential_marker_major", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for m in 0..n_markers {
                let row_offset = m * n_states;
                for k in 0..n_states {
                    sum += data[row_offset + k];
                }
            }
            black_box(sum)
        })
    });

    // State-major access (potentially worse cache behavior)
    group.bench_function("sequential_state_major", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for k in 0..n_states {
                for m in 0..n_markers {
                    sum += data[m * n_states + k];
                }
            }
            black_box(sum)
        })
    });

    group.finish();
}

/// Benchmark priority queue operations (like ImpStates uses)
fn bench_priority_queue(c: &mut Criterion) {
    use std::collections::BinaryHeap;

    let mut group = c.benchmark_group("priority_queue");

    for n_ops in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(n_ops as u64));

        group.bench_with_input(
            BenchmarkId::new("push_pop", n_ops),
            &n_ops,
            |b, &n_ops| {
                b.iter(|| {
                    let mut heap: BinaryHeap<i32> = BinaryHeap::with_capacity(n_ops);
                    // Simulate ImpStates pattern: push, then pop oldest
                    for i in 0..n_ops {
                        heap.push(i as i32);
                        if heap.len() > 1600 {
                            heap.pop();
                        }
                    }
                    black_box(heap.len())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark HashMap operations (like hap_to_last_ibs)
fn bench_hashmap_vs_vec(c: &mut Criterion) {
    use std::collections::HashMap;

    let mut group = c.benchmark_group("lookup_structure");

    let n_ref_haps = 32000; // Typical reference panel size
    let n_lookups = 100000;

    // Pre-populate
    let mut hashmap: HashMap<u32, i32> = HashMap::with_capacity(n_ref_haps);
    let mut vec_map: Vec<i32> = vec![i32::MIN; n_ref_haps];

    for i in 0..n_ref_haps {
        if i % 3 == 0 {
            hashmap.insert(i as u32, i as i32);
            vec_map[i] = i as i32;
        }
    }

    let lookup_keys: Vec<u32> = (0..n_lookups).map(|i| (i % n_ref_haps) as u32).collect();

    group.throughput(Throughput::Elements(n_lookups as u64));

    group.bench_function("hashmap_lookup", |b| {
        b.iter(|| {
            let mut sum = 0i32;
            for &key in &lookup_keys {
                sum = sum.wrapping_add(hashmap.get(&key).copied().unwrap_or(i32::MIN));
            }
            black_box(sum)
        })
    });

    group.bench_function("vec_lookup", |b| {
        b.iter(|| {
            let mut sum = 0i32;
            for &key in &lookup_keys {
                sum = sum.wrapping_add(vec_map[key as usize]);
            }
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fwd_update,
    bench_bwd_update,
    bench_forward_backward_scaling,
    bench_threaded_haps_traversal,
    bench_mismatch_computation,
    bench_memory_access_patterns,
    bench_priority_queue,
    bench_hashmap_vs_vec,
);

criterion_main!(benches);
