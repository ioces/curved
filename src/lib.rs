use ndarray::{s, Axis, Array1, ArrayViewMut1, ArrayView1, ArrayView2, CowArray, Ix1, Ix2, Slice};
use numpy::{convert::IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pymodule, pyfunction, wrap_pyfunction, Py, PyModule, PyResult, Python};


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


struct LineStartPointBuffer<'a> {
    vectors: CowArray<'a, f64, Ix2>,
    magnitudes_2: CowArray<'a, f64, Ix1>,
}


impl LineStartPointBuffer<'_> {
    fn from_points<'a>(points: ArrayView2<f64>) -> LineStartPointBuffer<'a> {
        let start = points.slice(s![0, ..]);
        let vectors = &points - &start;
        let magnitudes_2 = (&vectors * &vectors).sum_axis(Axis(1));
        LineStartPointBuffer { vectors: vectors.into(), magnitudes_2: magnitudes_2.into() }
    }

    fn subset(&self, slice: Slice) -> LineStartPointBuffer {
        LineStartPointBuffer {vectors: self.vectors.slice(s![slice, ..]).into(),
            magnitudes_2: self.magnitudes_2.slice(s![slice]).into()}
    }
}


fn line_point_distances_2(
    start: ArrayView1<'_, f64>,
    end: ArrayView1<'_, f64>,
    buffer: &LineStartPointBuffer<'_>
) -> Array1<f64> {
    // Calculate the unit vector from the start (A) to the end (B)
    let ab: Array1<f64> = &end - &start;
    let ab_magnitude = ab.dot(&ab).sqrt();
    let ab_unit = ab / ab_magnitude;

    // Project the point (C) to start (A) vector onto the unit vector using a dot product.
    // It should return the magnitude of the vector from the start (A) to the point closest
    // to the AB line (D).
    let ad_magnitudes_2 = buffer.vectors.dot(&ab_unit).mapv(|v| v.powi(2));
    &buffer.magnitudes_2 - &ad_magnitudes_2
}

fn rdp_recurse(points: ArrayView2<'_, f64>, buffer: Option<LineStartPointBuffer<'_>>, mut mask: ArrayViewMut1<'_, bool>, epsilon_2: f64) {
    // Get the start and end points of the curve
    let start = points.slice(s![0, ..]);
    let end = points.slice(s![-1, ..]);

    // Calculate some buffered line start to point values.
    let buffer = buffer.unwrap_or_else(|| LineStartPointBuffer::from_points(points));

    // Calculate the distance squared between the line and each point.
    let distances_2 = line_point_distances_2(start, end, &buffer);

    // Find the point with the maximum distance from the line joining the endpoints.
    let mut d_2_max = 0.0;
    let mut i_max: usize = 0;
    for (i, d) in distances_2.iter().enumerate().take(distances_2.len() - 1).skip(1) {
        if d > &d_2_max {
            i_max = i;
            d_2_max = *d;
        }
    }

    // If that point is further away from the line joining the endpoints than epsilon^2,
    // mark it as retained and recurse the algorithm for the line joining start->point
    // and point->end.
    if d_2_max > epsilon_2 {
        mask[i_max] = true;
        rdp_recurse(points.slice(s![..=i_max, ..]),
            Some(buffer.subset(Slice::from(..=i_max))),
            mask.slice_mut(s![..=i_max]),
            epsilon_2);
        rdp_recurse(points.slice(s![i_max.., ..]), None, mask.slice_mut(s![i_max..]), epsilon_2);
    }
}

#[pymodule]
fn _rustlib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfunction]
    fn rdp(
        py: Python<'_>,
        points: &PyArray2<f64>,
        epsilon: f64
    ) -> Py<PyArray1<bool>> {
        let points = points.readonly();
        crate::rdp(points.as_array(), epsilon).into_pyarray(py).to_owned()
    }

    m.add_function(wrap_pyfunction!(rdp, m)?)?;

    Ok(())
}
