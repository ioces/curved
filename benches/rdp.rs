#[macro_use]
extern crate criterion;

use criterion::{Criterion, BenchmarkId};
use pprof::criterion::{Output, PProfProfiler};

use ndarray::{Axis, Array1, Array2, concatenate, array};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use ndarray_stats::{QuantileExt};


fn rdp_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdp_2d");
    for size in [1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut y = Array2::random((size, 1), StandardNormal);
            y.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
            let max = (*y.max().unwrap() as f64).max((*y.min().unwrap() as f64).abs());
            let epsilon = max / 2000.0;
            let x = Array1::linspace(0.0, max, size).insert_axis(Axis(1));
            let points = concatenate![Axis(1), x, y];

            b.iter(|| {
                curved::rdp(points.view(), epsilon)
            });
        });
    }
    group.finish();

    c.bench_function("rdp_large_3d", |b| {
        let points = concatenate![
            Axis(1),
            Array1::range(0.0, 10000.0, 1.0).insert_axis(Axis(1)),
            Array2::random((10000, 2), StandardNormal)
        ];

        b.iter(|| {
            curved::rdp(points.view(), 0.1);
        });
    });

    c.bench_function("rdp_norway", |b| {
        let points = include!("../fixtures/norway_main.rs");

        b.iter(|| {
            curved::rdp(points.view(), 0.0005);
        });
    });
}

criterion_group!{
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = rdp_benches
}
criterion_main!(benches);