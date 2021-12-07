use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use pbr::ProgressBar;

#[derive(Debug)]
pub struct MLP {
    learning_rate: f64,
    wij: Array2<f64>,
    wjk: Array2<f64>,
}

impl MLP {
    pub fn new(input: usize, hidden: usize, output: usize, learning_rate: f64) -> MLP {
        MLP {
            learning_rate,
            wij: Array2::random((input, hidden), Uniform::new(-1., 1.)),
            wjk: Array2::random((hidden, output), Uniform::new(-1., 1.)),
        }
    }

    pub fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        iterations: i32,
    ) -> ArcArray<f64, Ix2> {
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
                .dot(&(&delta_y * tanh_derivative(&x_j.view(), &self.wjk.view())));

            // Gradients for input to hidden weights
            let g_wij = x_i.t().dot(
                &((&delta_y * tanh_derivative(&x_j.view(), &self.wjk.view()).dot(&self.wjk.t()))
                    * tanh_derivative(&x_i.view(), &self.wij.view())),
            );

            self.wij = &self.wij + (g_wij * self.learning_rate);
            self.wjk = &self.wjk + (g_wjk * self.learning_rate);
            pb.inc();
        }
        pb.finish_print("Finished Training");
        y_hat
    }

    pub fn predict(&self, x: &ArrayView2<f64>) -> f64 {
        let x_j = calculate_forward(x, &self.wij.view());
        *calculate_forward(&x_j.view(), &self.wjk.view())
            .get((0, 0))
            .unwrap()
    }
}

fn sigmoid(i: &f64) -> f64 {
    1.0 / (1.0 + (-i).exp())
}

fn tanh(i: &f64) -> f64 {
    i.tanh()
}

fn calculate_forward(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
    x.dot(y).map(tanh)
}

fn sigmoid_derivative(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
    let mut s = calculate_forward(&x, &y);
    let ones = Array2::ones(s.raw_dim());
    s = ones - s;
    calculate_forward(x, y) * s
}

fn tanh_derivative(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
    let mut s = calculate_forward(&x, &y);
    s = &s * &s.view();
    let ones = Array2::ones(s.raw_dim());
    ones - s
}
