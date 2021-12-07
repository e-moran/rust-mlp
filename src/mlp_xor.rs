use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use pbr::ProgressBar;

#[derive(Debug)]
pub struct MLP {
    learning_rate: f64,
    pub(crate) wij: Array2<f64>,
    wjk: Array2<f64>,
}

impl MLP {
    pub fn new(input: usize, hidden: usize, output: usize, learning_rate: f64) -> MLP {
        MLP {
            learning_rate,
            wij: Array2::random((input, hidden), Uniform::new(0., 1.)),
            wjk: Array2::random((hidden, output), Uniform::new(0., 1.)),
        }
    }

    pub fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        iterations: i32,
    ) -> Array2<i8> {
        let mut y_hat = Array2::zeros((1, 1)).into_shared();
        let mut pb = ProgressBar::new(iterations as u64);
        pb.format("╢▌▌░╟");
        for _ in 0..iterations {
            let x_i = x;
            let x_j = calculate_forward(x_i, &self.wij.view());
            y_hat = calculate_forward(&x_j.view(), &self.wjk.view()).into_shared();
            let delta_y: Array2<f64> = y - y_hat.to_owned();

            // Gradients for hidden to output weights
            let g_wjk = x_j
                .t()
                .dot(&(&delta_y * sigmoid_derivative(&x_j.view(), &self.wjk.view())));

            // Gradients for input to hidden weights
            let g_wij = x_i.t().dot(
                &((&delta_y
                    * sigmoid_derivative(&x_j.view(), &self.wjk.view()).dot(&self.wjk.t()))
                    * sigmoid_derivative(&x_i.view(), &self.wij.view())),
            );

            self.wij = &self.wij + g_wij;
            self.wjk = &self.wjk + g_wjk;
            pb.inc();
        }

        pb.finish_print("done");
        y_hat.map(to_int)
    }

    pub fn predict(&self, x: &ArrayView2<f64>) -> Array2<i8> {
        let x_j = calculate_forward(x, &self.wij.view());
        calculate_forward(&x_j.view(), &self.wjk.view())
            .map(to_int)
    }
}

fn to_int(x: &f64) -> i8 {
    x.round() as i8
}

fn sigmoid(i: &f64) -> f64 {
    1.0 / (1.0 + (-i).exp())
}

fn calculate_forward(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
    x.dot(y).map(sigmoid)
}

fn sigmoid_derivative(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
    let mut s = calculate_forward(&x, &y);
    let ones = Array2::ones(s.raw_dim());
    s = ones - s;
    calculate_forward(x, y) * s
}
