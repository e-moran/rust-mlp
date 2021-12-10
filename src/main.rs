mod metrics;
mod nn;

use csv::ReaderBuilder;

use ndarray::prelude::*;

use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use ndarray_csv::Array2Reader;

use std::fs::File;

fn main() {
    letter_recognition();
}

// PART 1 XOR:

#[allow(dead_code)]
fn xor() {
    let x: Array2<f64> = arr2(&[[0., 0., 1., 1.], [0., 1., 0., 1.]]);
    let y = array![[0., 1., 1., 0.]];

    println!("Input: {:?}", &x);

    let mut test_mlp = nn::NeuralNetwork::new(
        vec![2, 4, 1],
        nn::Activation::Sigmoid,
        nn::OptimizationAlgorithm::Batch,
        (0., 1.),
        false,
    );

    let test_2 = test_mlp.fit(&x.view(), &y.view(), 0.1, 50000);
    println!("MLP: {:#?}", &test_mlp);

    println!("\nGradient Descent: \n{:#?}", test_2);

    let xor_test_input = arr2(&[[1.], [1.]]);
    let xor_predict_test = test_mlp.predict(&xor_test_input.view());
    println!("Test error: {}", &xor_predict_test);
}

// PART 2 SIN:
#[allow(dead_code)]
fn sin() {
    let mut mlp = nn::NeuralNetwork::new(
        vec![4, 20, 20, 20, 20, 20, 1],
        nn::Activation::Sigmoid,
        nn::OptimizationAlgorithm::Batch,
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

    test_train_split(&x.view(), &y.view(), samples_for_test)
}

fn test_train_split(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    samples_for_test: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let (x_train, x_test) = x.split_at(Axis(1), x.ncols() - samples_for_test);
    let (y_train, y_test) = y.split_at(Axis(1), y.ncols() - samples_for_test);

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

// PART 3 LETTER RECOGNITION

fn letter_recognition() {
    let mut mlp = nn::NeuralNetwork::new(
        vec![16, 64, 32, 26],
        nn::Activation::Sigmoid,
        nn::OptimizationAlgorithm::Stochastic,
        (-1., 1.),
        false,
    );

    let (x, y) = read_letter_data();
    let (x_train, x_test, y_train, y_test) = test_train_split(&x.t(), &y.t(), 4000);

    mlp.fit(&x_train.view(), &y_train.view(), 0.1, 15);

    let y_pred = mlp.predict(&x_test.view());

    let mut it = y_test.axis_iter(Axis(1)).zip(y_pred.axis_iter(Axis(1)));
    let mut matched = 0;
    for _ in 0..y_pred.ncols() {
        let (a, b) = it.next().unwrap();
        if max_in_row(&a) == max_in_row(&b) {
            matched += 1;
        }
    }

    println!("Matched: {}", &matched);
}

fn read_letter_data() -> (Array2<f64>, Array2<f64>) {
    let file = File::open("letter-recognition.data").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let input_array: Array2<String> = reader.deserialize_array2_dynamic().unwrap();

    let (y_str, x_str) = input_array.view().split_at(Axis(1), 1);
    let x = x_str.map(|a| a.parse::<i32>().unwrap() as f64);
    let y_linear = y_str.map(|b| (b.chars().next().unwrap() as i8) as f64);

    let mut y: Array2<f64> = Array2::zeros((y_linear.nrows(), 26));
    for i in 0..y.nrows() {
        y[[i, ((y_linear[[i, 0]] - 65.) as usize)]] = 1.;
    }
    (x, y)
}

fn max_in_row(row: &ArrayView1<f64>) -> usize {
    let mut biggest: f64 = 0.;
    let mut biggest_index: usize = 0;
    for (i, x) in row.iter().enumerate() {
        if x > &biggest {
            biggest = *x;
            biggest_index = i;
        }
    }

    biggest_index
}
