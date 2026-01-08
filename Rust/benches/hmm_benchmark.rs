use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use reagle::model::hmm::HmmUpdater;

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

criterion_group!(
    benches,
    bench_fwd_update,
    bench_bwd_update,
    bench_forward_backward_scaling,
);

criterion_main!(benches);
