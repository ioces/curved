#[macro_use]
extern crate criterion;

use criterion::Criterion;
use ndarray::{Axis, Array1, Array2, concatenate, array};
use ndarray_rand::{RandomExt, rand_distr::Uniform};


fn rdp_benches(c: &mut Criterion) {
    c.bench_function("rdp_large_2d", |b| {
        let points = concatenate![
            Axis(1),
            Array1::range(0.0, 10000.0, 1.0).insert_axis(Axis(1)),
            Array2::random((10000, 1), Uniform::new(0.0, 1.0))
        ];

        b.iter(|| {
            curved::rdp(points.view(), 0.1);
        });
    });

    c.bench_function("rdp_large_3d", |b| {
        let points = concatenate![
            Axis(1),
            Array1::range(0.0, 10000.0, 1.0).insert_axis(Axis(1)),
            Array2::random((10000, 2), Uniform::new(0.0, 1.0))
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

criterion_group!(benches, rdp_benches);
criterion_main!(benches);