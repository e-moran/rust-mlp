mod metrics;
mod mlp;

use csv::ReaderBuilder;
use ndarray::prelude::*;
use ndarray_csv::Array2Reader;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::env;
use std::fs::File;

use crate::mlp::MLP;

enum Mode {
    Load,
    Train,
    TrainSave,
}

impl Mode {
    pub fn get_mode(mode_string: &str) -> Result<Mode, &str> {
        match mode_string {
            "load" => Ok(Mode::Load),
            "train" => Ok(Mode::Train),
            "train_save" => Ok(Mode::TrainSave),
            _ => Err("Invalid first argument. Options are load/train/train_save"),
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        panic!("You must specify two arguments.\nThe first can be load/train/train_save\nThe second can be xor/sin/letters");
    }

    let mode = match Mode::get_mode(args[1].as_str()) {
        Ok(m) => m,
        Err(e) => panic!("{}", e),
    };

    match args[2].as_str() {
        "xor" => xor(mode),
        "sin" => sin(mode),
        "letters" => letter_recognition(mode),
        _ => panic!("Invalid second argument. Options are xor/sin/letters"),
    }
}

// PART 1 XOR:

fn xor(mode: Mode) {
    let x: Array2<f64> = arr2(&[[0., 0., 1., 1.], [0., 1., 0., 1.]]);
    let y = array![[0., 1., 1., 0.]];

    let mut mlp: MLP = match mode {
        Mode::Load => mlp::MLP::from_file("xor_saved_mlp.json").unwrap_or_else(|e| panic!("{}", e)),
        _ => mlp::MLP::new(
            vec![2, 3, 1],
            mlp::Activation::Sigmoid,
            mlp::OptimizationAlgorithm::Batch,
            (0., 1.),
            false,
        ),
    };

    if !matches!(mode, Mode::Load) {
        mlp.fit(&x.view(), &y.view(), 0.5, 2500);
    }

    let xor_predict_test = mlp.predict(&x.view()).map(xor_round);
    println!("Test output: {}", &xor_predict_test);

    if matches!(mode, Mode::TrainSave) {
        match mlp.save_to_file("xor_saved_mlp.json") {
            Err(e) => println!("{}", e),
            _ => println!("XOR MLP Successfully Saved"),
        }
    }
}

fn xor_round(x: &f64) -> i8 {
    x.round() as i8
}

// PART 2 SIN:
fn sin(mode: Mode) {
    let mut mlp: MLP = match mode {
        Mode::Load => mlp::MLP::from_file("sin_saved_mlp.json").unwrap_or_else(|e| panic!("{}", e)),
        _ => mlp::MLP::new(
            vec![4, 25, 20, 1],
            mlp::Activation::Sigmoid,
            mlp::OptimizationAlgorithm::Batch,
            (-1., 1.),
            true,
        ),
    };

    let (x_train, x_test, y_train, y_test) = generate_for_sin(500, 100);

    if !matches!(mode, Mode::Load) {
        mlp.fit(&x_train.view(), &y_train.map(squash_sin).view(), 0.1, 60000);
    }

    let training_y_hat = mlp.predict(&x_train.view()).map(unsquash_sin);
    // Compute the error for the training set.
    let training_metrics =
        metrics::DistanceMetrics::generate_metrics(&y_train.view(), &training_y_hat.view());
    println!("\nTraining Set Error:\n{:#?}", &training_metrics);

    let test_y_hat = mlp.predict(&x_test.view()).map(unsquash_sin);

    // Compute the error for the test set.
    let test_metrics =
        metrics::DistanceMetrics::generate_metrics(&y_test.view(), &test_y_hat.view());
    println!("\nTest Set Error:\n{:#?}", &test_metrics);

    if matches!(mode, Mode::TrainSave) {
        match mlp.save_to_file("sin_saved_mlp.json") {
            Err(e) => println!("{}", e),
            _ => println!("Sine Approximation MLP Successfully Saved"),
        }
    }
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

fn squash_sin(i: &f64) -> f64 {
    (i + 1.) / 2.
}

fn unsquash_sin(i: &f64) -> f64 {
    (i * 2.) - 1.
}

// PART 3 LETTER RECOGNITION
fn letter_recognition(mode: Mode) {
    let mut mlp: MLP = match mode {
        Mode::Load => mlp::MLP::from_file("lr_saved_mlp.json").unwrap_or_else(|e| panic!("{}", e)),
        _ => mlp::MLP::new(
            vec![16, 256, 128, 64, 26],
            mlp::Activation::Sigmoid,
            mlp::OptimizationAlgorithm::Stochastic(0.95),
            (-1., 1.),
            false,
        ),
    };

    let (x, y) = read_letter_data();
    let (x_train, x_test, y_train, y_test) = test_train_split(&x.t(), &y.t(), 4000);

    if !matches!(mode, Mode::Load) {
        mlp.fit(&x_train.view(), &y_train.view(), 0.5, 64);
    }

    let y_pred = mlp.predict(&x_test.view());
    let matched = metrics::count_matched_letters(&y_test.view(), &y_pred.view());

    println!(
        "Matched: {}/4000 ({:.2}%)",
        &matched,
        (matched as f64 / 4000.) * 100.
    );

    if matches!(mode, Mode::TrainSave) {
        match mlp.save_to_file("lr_saved_mlp.json") {
            Err(e) => println!("{}", e),
            _ => println!("Letter Recognition MLP Successfully Saved"),
        }
    }
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

// GENERIC
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
