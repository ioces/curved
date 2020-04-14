use ndarray::{array};

#[test]
fn rdp_norway() {
    let points = include!("../fixtures/norway_main.rs");
    let comparison = include!("../fixtures/norway_mask_0.0005.rs");
    let mask = curved::rdp(points.view(), 0.0005);
    assert_eq!(mask, comparison);
}