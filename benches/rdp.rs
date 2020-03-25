#![feature(test)]

extern crate test;

use test::{Bencher};
use ndarray::{Axis, Array1, Array2, stack};
use ndarray_rand::{RandomExt, rand_distr::Uniform};


#[bench]
fn large_2d(b: &mut Bencher) {
    let points = stack![
        Axis(1),
        Array1::range(0.0, 20000.0, 1.0).insert_axis(Axis(1)),
        Array2::random((20000,1), Uniform::new(0.0, 1.0))
    ];

    b.iter(|| {
        curved::rdp(points.view(), 0.1);
    })
}