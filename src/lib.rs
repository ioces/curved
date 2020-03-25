#![deny(rust_2018_idioms)]
use ndarray::{s, Axis, Array1, Array2, ArrayViewMut1, ArrayView2};
use numpy::{convert::IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};

pub fn rdp(points: ArrayView2<'_, f64>, epsilon: f64) -> Array1<bool> {
    // Generate a mask boolean array, which will be the result.
    let mut mask = Array1::from_elem((points.len_of(Axis(0)),), false);
    mask[0] = true;
    let mask_len = mask.len();
    mask[mask_len - 1] = true;

    // Run the recursive RDP algorithm
    rdp_recurse(points, mask.view_mut(), epsilon.powi(2));

    mask
}

fn rdp_recurse(points: ArrayView2<'_, f64>, mut mask: ArrayViewMut1<'_, bool>, epsilon_2: f64) {
    // Get the start and end points of the curve
    let start = points.slice(s![0, ..]);
    let end = points.slice(s![-1, ..]);

    // Find distances from each point to the line joining the endpoints.
    let mut unit: Array1<f64> = &end - &start;
    unit /= unit.dot(&unit).sqrt();
    let end_mids: Array2<f64> = &end.broadcast(points.raw_dim()).unwrap() - &points;
    let end_mid_2 = (&end_mids * &end_mids).sum_axis(Axis(1));
    let end_mid_unit = unit.dot(&end_mids.reversed_axes());
    let distances: Array1<f64> = end_mid_2 - &end_mid_unit * &end_mid_unit;

    // Find the point with the maximum distance from the line joining the endpoints.
    let mut d_max = 0.0;
    let mut i_max: usize = 0;
    for (i, d) in distances.iter().enumerate().take(distances.len() - 1).skip(1) {
        if d > &d_max {
            i_max = i;
            d_max = *d;
        }
    }

    // If that point is further away from the line joining the endpoints than epsilon^2,
    // mark it as retained and recurse the algorithm for the line joining start->point
    // and point->end.
    if d_max > epsilon_2 {
        mask[i_max] = true;
        rdp_recurse(points.slice(s![..=i_max, ..]), mask.slice_mut(s![..=i_max]), epsilon_2);
        rdp_recurse(points.slice(s![i_max.., ..]), mask.slice_mut(s![i_max..]), epsilon_2);
    }
}

#[pymodule]
fn _rustlib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "rdp")]
    fn rdp(
        py: Python<'_>,
        points: &PyArray2<f64>,
        epsilon: f64
    ) -> Py<PyArray1<bool>> {
        let points = points.as_array();
        crate::rdp(points, epsilon).into_pyarray(py).to_owned()
    }

    Ok(())
}
