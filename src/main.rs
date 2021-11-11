use ndarray::{arr1, arr2, array};
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
    let x = array![[0,0], [0,1], [1,0], [1,1]];
    let y = array![[0], [1], [1], [0]];
    let set = TrainingSet {
        x: arr2(&[[0,0], [0,1], [1,0], [1,1]]),
        y: arr1(&[0, 1, 1, 0])
    };
    println!("{:?} {:?} {:?}", x, y, set);
}
