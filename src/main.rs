mod mlp;
mod mlp_xor;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::fmt;
use std::fmt::Formatter;

struct TrainingSet {
    x: ndarray::Array2<i32>,
    y: ndarray::Array1<i32>,
}

impl fmt::Debug for TrainingSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "x: {:?}\ny: {:?}", self.x, self.y)
    }
}

fn main() {
    sin();
}

fn xor() {
    let x: Array2<f64> = arr2(&[[0., 0.], [0., 1.], [1., 0.], [1., 1.]]);
    let y = array![[0.], [1.], [1.], [0.]];

    println!("Input: {:?}", &x);

    let mut test_mlp = mlp_xor::MLP::new(2, 4, 1, 0.1);
    println!("{:?}", &test_mlp.wij);

    let test_2 = test_mlp.fit(&x.view(), &y.view(), 10000);
    println!("Gradient Descent: \n{:?}", test_2);

    let xor_test_input = arr2(&[[1., 1.]]);
    let xor_predict_test = test_mlp.predict(&xor_test_input.view());
    println!("Test output: {}", &xor_predict_test);
}

fn sin() {
    let mut test_mlp = mlp::MLP::new(4, 128, 1, 0.001);

    let mut x: Array2<f64> = Array2::random((500, 4), Uniform::new(-1., 1.));
    let y = Array::from_iter(
        x.axis_iter(Axis(0))
            .map(|x| (x[0] - x[1] + x[2] - x[3]).sin()),
    )
    .insert_axis(Axis(1)); // Convert to 2d.

    let test_2 = test_mlp.fit(&x.view(), &y.view(), 1000);
}
