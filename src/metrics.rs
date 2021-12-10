use ndarray::prelude::*;

#[derive(Debug)]
pub struct DistanceMetrics {
    mean: f64,
    max: f64,
    min: f64,
}

impl DistanceMetrics {
    pub fn generate_metrics(y: &ArrayView2<f64>, y_hat: &ArrayView2<f64>) -> DistanceMetrics {
        assert_eq!(
            y.shape(),
            y_hat.shape(),
            "y and y_hat must have the same shape for metrics."
        );

        let mut y_delta: Vec<f64> = y
            .axis_iter(Axis(1))
            .zip(y_hat.axis_iter(Axis(1)))
            .map(|(a, b)| (a[0] - b[0]).abs())
            .collect();

        let mean = y_delta.iter().sum::<f64>() / y_delta.len() as f64;
        y_delta.sort_by(|a, b| a.partial_cmp(b).unwrap());

        DistanceMetrics {
            mean,
            max: y_delta[y_delta.len() - 1],
            min: y_delta[0],
        }
    }
}
