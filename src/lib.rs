#![deny(rust_2018_idioms)]
use ndarray::{s, stack, Axis, Array1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};

#[pymodule]
fn _rustlib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn rdp(points: ArrayView2<'_, f64>, epsilon: &f64) -> Array1<bool> {
        if points.is_empty() {
            return Array1::default((0,));
        }
        
        // TODO - Maybe creating a Line struct would make sense rather than this math here.
        
        let start = points.slice(s![0, ..]);
        let end = points.slice(s![-1, ..]);

        // Calculate the unit vector between the start and end points.
        let mut unit: Array1<f64> = &end - &start;
        unit /= unit.dot(&unit).sqrt();
        
        let mut result = Array1::zeros((points.len_of(Axis(0)),));

        // Find the point with the maximum distance from the line between the two endpoints.
        let mut d_max = 0.0;
        let mut i_max: usize = 0;
        let mut distance: f64;
        for (i, mid) in points.outer_iter().enumerate().take(points.len_of(Axis(0)) - 1).skip(1) {
            let end_mid = &end - &mid;
            distance = (end_mid.dot(&end_mid) - end_mid.dot(&unit).powi(2)).sqrt();
            if distance > d_max {
                i_max = i;
                d_max = distance;
            }
            result[i] = distance;
        }

        if d_max > *epsilon {
            let head = rdp(points.slice(s![..=i_max, ..]), epsilon);
            let tail = rdp(points.slice(s![i_max.., ..]), epsilon);
            stack(Axis(0), &[head.slice(s![..-1]), tail.view()]).unwrap()
        } else {
            let result_length = points.len_of(Axis(0));
            let mut result = Array1::from_elem((result_length,), false);
            result[0] = true;
            result[result_length - 1] = true;
            result
        }
    }

    #[pyfn(m, "rdp")]
    fn rdp_py(
        py: Python<'_>,
        points: &PyArray2<f64>,
        epsilon: f64
    ) -> Py<PyArray1<bool>> {
        let points = points.as_array();
        rdp(points, &epsilon).into_pyarray(py).to_owned()
    }

    Ok(())
}
