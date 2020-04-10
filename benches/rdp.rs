#![feature(test)]

extern crate test;

use test::{Bencher};
use ndarray::{Axis, Array1, Array2, stack, array};
use ndarray_rand::{RandomExt, rand_distr::Uniform};


#[bench]
fn rdp_large_2d(b: &mut Bencher) {
    let points = stack![
        Axis(1),
        Array1::range(0.0, 10000.0, 1.0).insert_axis(Axis(1)),
        Array2::random((10000, 1), Uniform::new(0.0, 1.0))
    ];

    b.iter(|| {
        curved::rdp(points.view(), 0.1);
    })
}

#[bench]
fn rdp_large_3d(b: &mut Bencher) {
    let points = stack![
        Axis(1),
        Array1::range(0.0, 10000.0, 1.0).insert_axis(Axis(1)),
        Array2::random((10000, 2), Uniform::new(0.0, 1.0))
    ];

    b.iter(|| {
        curved::rdp(points.view(), 0.1);
    })
}

#[bench]
fn rdp_norway(b: &mut Bencher) {
    let points = include!("../fixtures/norway_main.rs");

    b.iter(|| {
        curved::rdp(points.view(), 0.0005);
    })
}