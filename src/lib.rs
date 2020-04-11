use ndarray::{s, Axis, Array1, Array2, ArrayViewMut1, ArrayView1, ArrayView2, CowArray, Ix1, Ix2};
use numpy::{convert::IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};


struct Buffer<'a>(CowArray<'a, f64, Ix2>, CowArray<'a, f64, Ix1>);


pub fn rdp(points: ArrayView2<'_, f64>, epsilon: f64) -> Array1<bool> {
    // Generate a mask boolean array, which will be the result.
    let mut mask = Array1::from_elem((points.len_of(Axis(0)),), false);
    mask[0] = true;
    let mask_len = mask.len();
    mask[mask_len - 1] = true;

    // Run the recursive RDP algorithm
    rdp_recurse(points, None, mask.view_mut(), epsilon.powi(2));

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

fn line_point_distances_3(
    start: ArrayView1<'_, f64>,
    end: ArrayView1<'_, f64>,
    cas: ArrayView2<'_, f64>,
    mag_cas_2: ArrayView1<'_, f64>,
    points: ArrayView2<'_, f64>
) -> Array1<f64> {
    let ab: Array1<f64> = &end - &start;
    let ab_mag = ab.dot(&ab).sqrt();
    let ab_u = ab / ab_mag;

    let mag_ads_2 = cas.dot(&ab_u).mapv(|v| v.powi(2));
    &mag_cas_2 - &mag_ads_2
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

fn rdp_recurse(points: ArrayView2<'_, f64>, opt_buffer: Option<Buffer>, mut mask: ArrayViewMut1<'_, bool>, epsilon_2: f64) {
    // Get the start and end points of the curve
    let start = points.slice(s![0, ..]);
    let end = points.slice(s![-1, ..]);

    let buffer = match opt_buffer {
        Some(b) => b,
        None => {
            let acs = &points - &end;
            let mag_acs_2 = (&acs * &acs).sum_axis(Axis(1));
            Buffer(acs.into(), mag_acs_2.into())
        }
    };
    let distances = line_point_distances_3(start, end, buffer.0.view(), buffer.1.view(), points);

    /*
    let distances = line_point_distances_3(start, end, points);
    */

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
        /*rdp_recurse(points.slice(s![..=i_max, ..]),
            Some(Buffer(buffer.0.slice(s![..=i_max, ..]).into(), buffer.1.slice(s![..=i_max]).into())),
            mask.slice_mut(s![..=i_max]),
            epsilon_2);*/
        rdp_recurse(points.slice(s![..=i_max, ..]), None, mask.slice_mut(s![..=i_max]), epsilon_2);
        rdp_recurse(points.slice(s![i_max.., ..]),
            Some(Buffer(buffer.0.slice(s![i_max.., ..]).into(), buffer.1.slice(s![i_max..]).into())),
            mask.slice_mut(s![i_max..]),
            epsilon_2);
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
