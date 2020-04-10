use ndarray::{s, Axis, Array1, Array2, ArrayViewMut1, ArrayView1, ArrayView2, array};
use numpy::{convert::IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};

pub fn rdp(points: ArrayView2<'_, f64>, epsilon: f64) -> Array1<bool> {
    // Generate a mask boolean array, which will be the result.
    let mut mask = Array1::from_elem((points.len_of(Axis(0)),), false);
    mask[0] = true;
    let mask_len = mask.len();
    mask[mask_len - 1] = true;

    // Run the recursive RDP algorithm
    rdp_recurse(points, mask.view_mut(), epsilon);

    mask
}

fn line_point_distances(start: ArrayView1<'_, f64>, end: ArrayView1<'_, f64>, points: ArrayView2<'_, f64>) -> Array1<f64> {
    let mut unit: Array1<f64> = &end - &start;
    unit /= unit.dot(&unit).sqrt();
    let end_mids: Array2<f64> = &end.broadcast(points.raw_dim()).unwrap() - &points;
    let end_mid_2 = (&end_mids * &end_mids).sum_axis(Axis(1));
    let end_mid_unit = unit.dot(&end_mids.reversed_axes());
    
    (end_mid_2 - &end_mid_unit * &end_mid_unit).mapv(f64::sqrt)
}

fn line_point_distances_3(start: ArrayView1<'_, f64>, end: ArrayView1<'_, f64>, points: ArrayView2<'_, f64>) -> Array1<f64> {
    let ab: Array1<f64> = &end - &start;
    let ab_mag = ab.dot(&ab).sqrt();
    let ab_u = ab / ab_mag;

    let cas: Array2<f64> = &start.broadcast(points.raw_dim()).unwrap() - &points;
    let mag_cas_2 = (&cas * &cas).sum_axis(Axis(1));
    let mag_ads_2 = ab_u.dot(&cas.reversed_axes()).mapv(|v| v.powi(2));
    (mag_cas_2 - mag_ads_2).mapv(f64::sqrt)
}

fn line_point_distances_4(start: ArrayView1<'_, f64>, end: ArrayView1<'_, f64>, points: ArrayView2<'_, f64>) -> Array1<f64> {
    let ab = &end - &start;
    let ab_mag = ab.dot(&ab).sqrt();
    let ab_u = ab / ab_mag;

    points.map_axis(Axis(1), |point| {
        let e = &end - &point;
        (e.dot(&e) - ab_u.dot(&e).powi(2)).sqrt()
    })
}

fn line_point_distances_2(start: ArrayView1<'_, f64>, end: ArrayView1<'_, f64>, points: ArrayView2<'_, f64>) -> Array1<f64> {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let sy = start[1];
    let sx = start[0];

    points.map_axis(Axis(1), |point| {
        let s = ((sy - point[1]) * dx - (sx - point[0]) * dy) / (dx.powi(2) + dy.powi(2));
        s.abs() * dx.hypot(dy)
    })
}

fn rdp_recurse(points: ArrayView2<'_, f64>, mut mask: ArrayViewMut1<'_, bool>, epsilon: f64) {
    // Get the start and end points of the curve
    let start = points.slice(s![0, ..]);
    let end = points.slice(s![-1, ..]);

    let distances = line_point_distances_4(start, end, points);

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
    if d_max > epsilon {
        mask[i_max] = true;
        rdp_recurse(points.slice(s![..=i_max, ..]), mask.slice_mut(s![..=i_max]), epsilon);
        rdp_recurse(points.slice(s![i_max.., ..]), mask.slice_mut(s![i_max..]), epsilon);
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
