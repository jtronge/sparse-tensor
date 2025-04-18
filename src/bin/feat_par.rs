//! Parallel tensor feature analysis.
//!
//! Partially based on https://arxiv.org/abs/2405.04944
use sparse_tensor::{load_tensor, get_tensor_dims};
use sparse_tensor::feat;
use std::fs::File;

fn main() {
    let args: Vec<String> = std::env::args().map(|arg| arg.to_string()).collect();
    assert_eq!(args.len(), 2);
    let tensor_fname = &args[1];
    println!("loading tensor {}", tensor_fname);
    let file = File::open(tensor_fname).expect("failed to open tensor");
    let tensor = load_tensor(file);
    feat::analyze_tensor(&tensor);
}
