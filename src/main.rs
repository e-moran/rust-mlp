mod metrics;
mod nn;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    sin();
}

#[allow(dead_code)]
fn xor() {
    let x: Array2<f64> = arr2(&[[0., 0., 1., 1.], [0., 1., 0., 1.]]);
    let y = array![[0., 1., 1., 0.]];

    println!("Input: {:?}", &x);

    let mut test_mlp =
        nn::NeuralNetwork::new(vec![2, 4, 1], nn::Activation::Sigmoid, (0., 1.), false);

    let test_2 = test_mlp.fit(&x.view(), &y.view(), 0.1, 50000);
    println!("MLP: {:#?}", &test_mlp);

    println!("\nGradient Descent: \n{:#?}", test_2);

    let xor_test_input = arr2(&[[1.], [1.]]);
    let xor_predict_test = test_mlp.predict(&xor_test_input.view());
    println!("Test error: {}", &xor_predict_test);
}

fn sin() {
    let mut mlp = nn::NeuralNetwork::new(
        vec![4, 20, 20, 20, 20, 20, 1],
        nn::Activation::Sigmoid,
        (-1., 1.),
        true,
    );

    let (x_train, x_test, y_train, y_test) = generate_for_sin(500, 100);

    let training_y_hat = mlp
        .fit(&x_train.view(), &y_train.map(squash_sin).view(), 0.1, 500)
        .unwrap()
        .map(unsquash_sin);

    // Compute the error for the training set.
    let training_metrics =
        metrics::DistanceMetrics::generate_metrics(&y_train.view(), &training_y_hat.view());
    println!("\nTraining Set Error:\n{:#?}", &training_metrics);

    let test_y_hat = mlp.predict(&x_test.view()).map(unsquash_sin);

    // Compute the error for the training set.
    let test_training_metrics =
        metrics::DistanceMetrics::generate_metrics(&y_test.view(), &test_y_hat.view());
    println!("\nTest Set Error:\n{:#?}", &test_training_metrics);
}

fn generate_for_sin(
    n_samples: usize,
    samples_for_test: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let x: Array2<f64> = Array2::random((4, n_samples), Uniform::new(-1., 1.));
    let y = Array::from_iter(
        x.axis_iter(Axis(1))
            .map(|x| (x[0] - x[1] + x[2] - x[3]).sin()),
    )
    .insert_axis(Axis(0)); // Convert to 2d.

    let (x_train, x_test) = x.view().split_at(Axis(1), n_samples - samples_for_test);
    let (y_train, y_test) = y.view().split_at(Axis(1), n_samples - samples_for_test);

    (
        x_train.to_owned(),
        x_test.to_owned(),
        y_train.to_owned(),
        y_test.to_owned(),
    )
}

fn squash_sin(i: &f64) -> f64 {
    (i + 1.) / 2.
}

fn unsquash_sin(i: &f64) -> f64 {
    (i * 2.) - 1.
}
