//! Tensor feature analysis.
//!
//! Partially based on https://arxiv.org/abs/2405.04944
use sparse_tensor::{load_tensor, get_tensor_dims};
use std::fs::File;

/// Calculate and return the mean and standard deviation of an array of counts.
fn calculate_mean_std_dev(counts: &[usize]) -> (f64, f64) {
    let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    let squared_sum = counts
        .iter()
        .map(|count| count * count)
        .sum::<usize>();
    let std_dev = (squared_sum as f64 / counts.len() as f64 - mean * mean).sqrt();
    (mean, std_dev)
}

fn main() {
    let args: Vec<String> = std::env::args().map(|arg| arg.to_string()).collect();
    assert_eq!(args.len(), 2);
    let tensor_fname = &args[1];
    println!("loading tensor {}", tensor_fname);
    let file = File::open(tensor_fname).expect("failed to open tensor");
    let tensor = load_tensor(file);
    let dims = get_tensor_dims(&tensor);
    assert_eq!(dims.len(), 3);

    // Organize into slices
    let mut slices = vec![vec![]; dims[0]];
    for (co, _) in &tensor {
        slices[co[0]].push(co.to_vec());
    }
    println!("nnz: {}", tensor.len());
    println!("density: {:e}", (tensor.len() as f64 / dims.iter().product::<usize>() as f64));

    // Get properties for each slice
    let mut nonzero_fibers_per_slice = vec![];
    let mut square_nnz_per_fiber_sum = 0;
    for i in 0..dims[0] {
        // X(i, :, k) fibers
        let mut fibers = vec![0; dims[2]];
        for co in &slices[i] {
            fibers[co[2]] += 1;
        }
        square_nnz_per_fiber_sum += fibers.iter().map(|count| count * count).sum::<usize>();
        let nonzero_fibers: usize = fibers.iter().map(|count| if *count > 0 { 1 } else { 0 }).sum();
        nonzero_fibers_per_slice.push(nonzero_fibers);
    }
    // Nonzero per fiber properties (X(i, :, k) fibers)
    let nnz_per_fiber_mean = tensor.len() as f64 / (dims[0] * dims[2]) as f64;
    let nnz_per_fiber_std_dev = (square_nnz_per_fiber_sum as f64 / tensor.len() as f64
                                 - nnz_per_fiber_mean * nnz_per_fiber_mean).sqrt();
    println!("nnz_per_fiber_mean: {:e}", nnz_per_fiber_mean);
    println!("nnz_per_fiber_std_dev: {}", nnz_per_fiber_std_dev);
    // Coefficient of variation
    println!("nnz_per_fiber_cv: {}", nnz_per_fiber_std_dev / nnz_per_fiber_mean);
    // Fiber per slice properties
    let fibers_per_slice_max = *nonzero_fibers_per_slice.iter().max().expect("missing maximum value");
    println!("fiber_density: {:e}", nonzero_fibers_per_slice.iter().sum::<usize>() as f64 / (dims[0] * dims[2]) as f64);
    let (fibers_per_slice_mean, fibers_per_slice_std_dev) = calculate_mean_std_dev(&nonzero_fibers_per_slice);
    println!("## statistics for X(i, :, k) fibers");
    println!("fibers_per_slice_mean: {:.4}", fibers_per_slice_mean);
    println!("fibers_per_slice_std_dev: {:.4}", fibers_per_slice_std_dev);
    // Coefficient of variation
    println!("fibers_per_slice_cv: {:.4}", fibers_per_slice_std_dev / fibers_per_slice_mean);
    // Imbalance calculated as (max - average) / max
    let fibers_per_slice_imbalance = (fibers_per_slice_max as f64 - fibers_per_slice_mean) / fibers_per_slice_max as f64;
    println!("fibers_per_slice_imbalance: {:.4}", fibers_per_slice_imbalance);
}
