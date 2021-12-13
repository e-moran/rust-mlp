use std::{fs, ops::Div};

use ndarray::prelude::*;
use ndarray_rand::rand_distr::num_traits::Pow;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use pbr::ProgressBar;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
struct LinearCache {
    a: Array2<f64>,
    w: Array2<f64>,
    b: Array2<f64>,
}

#[derive(Debug)]
struct ActivationCache {
    z: Array2<f64>,
}

#[derive(Debug)]
struct Cache {
    linear_cache: LinearCache,
    activation_cache: ActivationCache,
}

#[derive(Debug)]
struct Gradient {
    da: Array2<f64>,
    dw: Array2<f64>,
    db: Array2<f64>,
}

impl Gradient {
    pub fn from_tuple(grad: (Array2<f64>, Array2<f64>, Array2<f64>)) -> Gradient {
        Gradient {
            da: grad.0,
            dw: grad.1,
            db: grad.2,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
pub enum Activation {
    Sigmoid,
    Tanh,
    Relu,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
pub enum OptimizationAlgorithm {
    Batch,
    Stochastic(f64),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MLP {
    w: Vec<Array2<f64>>,
    b: Vec<Array2<f64>>,
    activation: Activation,
    optimization_algorithm: OptimizationAlgorithm,
    biases_enabled: bool,
}

impl MLP {
    pub fn new(
        layers: Vec<usize>,
        activation: Activation,
        optimization_algorithm: OptimizationAlgorithm,
        weights_range: (f64, f64),
        biases_enabled: bool,
    ) -> MLP {
        MLP {
            w: layers
                .iter()
                .enumerate()
                .filter(|&(i, _)| i < layers.len() - 1)
                .map(|(i, x)| rand_array((layers[i + 1], *x), weights_range))
                .collect(),
            b: layers
                .iter()
                .skip(1)
                .map(|x| Array2::zeros((*x, 1)))
                .collect(),
            activation,
            optimization_algorithm,
            biases_enabled,
        }
    }

    pub fn from_file(filename: &str) -> Result<MLP, String> {
        match fs::read_to_string(filename) {
            Ok(t) => serde_json::from_str(&t).map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn save_to_file(&self, filename: &str) -> Result<(), String> {
        match serde_json::to_string(self) {
            Ok(t) => fs::write(filename, t).map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        learning_rate: f64,
        iterations: i64,
    ) -> Option<Array2<f64>> {
        match self.optimization_algorithm {
            OptimizationAlgorithm::Batch => self.fit_batch(x, y, learning_rate, iterations),
            OptimizationAlgorithm::Stochastic(decay) => {
                self.fit_stochastic(x, y, learning_rate, decay, iterations)
            }
        }
    }

    fn fit_stochastic(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        mut learning_rate: f64,
        learning_rate_decay: f64,
        iterations: i64,
    ) -> Option<Array2<f64>> {
        let mut pb = ProgressBar::new((iterations * x.ncols() as i64) as u64);
        pb.format("╢▌▌░╟");

        let mut out: Option<Array2<f64>> = None;
        for _ in 0..iterations {
            for (sample_x, sample_y) in x.axis_iter(Axis(1)).zip(y.axis_iter(Axis(1))) {
                pb.inc();
                let (y_hat, caches) = self.model_forward(&sample_x.insert_axis(Axis(1)));
                let gradients =
                    self.model_backward(&y_hat.view(), &sample_y.insert_axis(Axis(1)), caches);
                self.update_parameters(gradients, learning_rate);
                out = Some(y_hat);
            }
            learning_rate *= learning_rate_decay;
        }
        pb.finish_println("Finished training.\n");
        out
    }

    fn fit_batch(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        learning_rate: f64,
        iterations: i64,
    ) -> Option<Array2<f64>> {
        let mut pb = ProgressBar::new(iterations as u64);
        pb.format("╢▌▌░╟");

        let mut out: Option<Array2<f64>> = None;
        for _ in 0..iterations {
            pb.inc();
            let (y_hat, caches) = self.model_forward(x);
            let gradients = self.model_backward(&y_hat.view(), y, caches);
            self.update_parameters(gradients, learning_rate);
            out = Some(y_hat);
        }
        pb.finish_println("Finished training.\n");
        out
    }

    #[allow(dead_code)]
    pub fn predict(&self, x: &ArrayView2<f64>) -> Array2<f64> {
        self.model_forward(x).0
    }

    fn model_forward(&self, x: &ArrayView2<f64>) -> (Array2<f64>, Vec<Cache>) {
        let mut y_hat = x.to_owned();
        let mut caches: Vec<Cache> = vec![];
        for (w, b) in self.w.iter().zip(self.b.iter()) {
            let (a, cache) = self.linear_activation_forward(&y_hat.view(), &w.view(), &b.view());
            y_hat = a;
            caches.push(cache);
        }
        (y_hat, caches)
    }

    fn update_parameters(&mut self, gradients: Vec<Gradient>, learning_rate: f64) {
        for (i, g) in gradients.iter().rev().enumerate() {
            self.w[i] = &self.w[i] - (&g.dw * learning_rate);
            if self.biases_enabled {
                self.b[i] = &self.b[i] - (&g.db * learning_rate);
            }
        }
    }

    fn model_backward(
        &self,
        y_hat: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        caches: Vec<Cache>,
    ) -> Vec<Gradient> {
        let mut gradients: Vec<Gradient> = vec![];

        for (i, cache) in caches.iter().rev().enumerate() {
            if i == 0 {
                gradients.push(Gradient::from_tuple(
                    self.linear_activation_backward(&loss(y_hat, y).view(), cache),
                ));
            } else {
                gradients.push(Gradient::from_tuple(self.linear_activation_backward(
                    &gradients.last().unwrap().da.view(),
                    cache,
                )));
            }
        }

        gradients
    }

    fn linear_activation_backward(
        &self,
        da: &ArrayView2<f64>,
        cache: &Cache,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let dz = match self.activation {
            Activation::Sigmoid => sigmoid_backward(da, &cache.activation_cache),
            Activation::Tanh => tanh_backward(da, &cache.activation_cache),
            Activation::Relu => relu_backward(da, &cache.activation_cache),
        };
        linear_backward(&dz.view(), &cache.linear_cache)
    }

    fn linear_activation_forward(
        &self,
        a_prev: &ArrayView2<f64>,
        w: &ArrayView2<f64>,
        b: &ArrayView2<f64>,
    ) -> (Array2<f64>, Cache) {
        let (z, linear_cache) = linear_forward(a_prev, w, b);
        let (a, activation_cache) = match self.activation {
            Activation::Sigmoid => sigmoid_forward(&z.view()),
            Activation::Tanh => tanh_forward(&z.view()),
            Activation::Relu => relu_forward(&z.view()),
        };
        return (
            a,
            Cache {
                linear_cache,
                activation_cache,
            },
        );
    }
}

fn linear_backward(
    dz: &ArrayView2<f64>,
    cache: &LinearCache,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let m = 1. / cache.a.nrows() as f64;
    let dw = m * dz.dot(&cache.a.t());
    let db = m * dz.sum_axis(Axis(1)).insert_axis(Axis(1));
    let da_prev = cache.w.t().dot(dz);

    (da_prev, dw, db)
}

fn linear_forward(
    a: &ArrayView2<f64>,
    w: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
) -> (Array2<f64>, LinearCache) {
    let cache = LinearCache {
        a: a.to_owned(),
        w: w.to_owned(),
        b: b.to_owned(),
    };
    (w.dot(a) + b, cache)
}

fn rand_array(shape: (usize, usize), weights_range: (f64, f64)) -> Array2<f64> {
    Array2::random(shape, Uniform::new(weights_range.0, weights_range.1))
}

fn sigmoid_backward(da: &ArrayView2<f64>, cache: &ActivationCache) -> Array2<f64> {
    let s = cache.z.map(sigmoid);
    da * s.map(sigmoid_derivative)
}

fn tanh_backward(da: &ArrayView2<f64>, cache: &ActivationCache) -> Array2<f64> {
    let s = cache.z.map(tanh);
    da * s.map(tanh_derivative)
}

fn relu_backward(da: &ArrayView2<f64>, cache: &ActivationCache) -> Array2<f64> {
    let s = cache.z.map(relu);
    da * s.map(relu_derivative)
}

fn sigmoid_forward(z: &ArrayView2<f64>) -> (Array2<f64>, ActivationCache) {
    let a = z.map(sigmoid);
    (a, ActivationCache { z: z.to_owned() })
}

fn tanh_forward(z: &ArrayView2<f64>) -> (Array2<f64>, ActivationCache) {
    let a = z.map(tanh);
    (a, ActivationCache { z: z.to_owned() })
}

fn relu_forward(z: &ArrayView2<f64>) -> (Array2<f64>, ActivationCache) {
    let a = z.map(relu);
    (a, ActivationCache { z: z.to_owned() })
}

fn sigmoid_derivative(x: &f64) -> f64 {
    x * (1. - x)
}

fn sigmoid(i: &f64) -> f64 {
    1. / (1. + (-i).exp())
}

fn tanh_derivative(i: &f64) -> f64 {
    1. - (i.tanh().pow(2))
}

fn tanh(i: &f64) -> f64 {
    i.tanh()
}

fn relu_derivative(i: &f64) -> f64 {
    if i > &0. {
        1.
    } else {
        0.
    }
}

fn relu(i: &f64) -> f64 {
    i.max(0.)
}

fn sub_one(i: &f64) -> f64 {
    1.0 - i
}

fn loss(y_hat: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
    (y.div(y_hat.to_owned()) - y.map(sub_one).div(y_hat.map(sub_one))) * -1.
}
