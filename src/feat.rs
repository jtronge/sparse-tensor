//! Tensor feature analysis and helping library.
use std::collections::BTreeMap;
use mpi::traits::*;
use mpi::collective::SystemOperation;
use crate::{Coordinate, get_tensor_dims, get_tensor_dims_iter};

/// Calculate and return the mean and standard deviation of an array of counts.
pub fn calculate_mean_std_dev(counts: &[usize]) -> (f64, f64) {
    let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    let squared_sum = counts
        .iter()
        .map(|count| count * count)
        .sum::<usize>();
    let std_dev = (squared_sum as f64 / counts.len() as f64 - mean * mean).sqrt();
    (mean, std_dev)
}

pub fn analyze_tensor(tensor: &BTreeMap<Vec<usize>, f64>) {
    let dims = get_tensor_dims(tensor);
    assert_eq!(dims.len(), 3);

    // Organize into slices
    let mut slices = vec![vec![]; dims[0]];
    for (co, _) in tensor {
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

pub fn analyze_tensor_parallel<P, C>(tensor_iter: impl Iterator<Item = (C, f64)> + Clone, comm: &P)
where
    P: AnyCommunicator,
    C: Coordinate,
{
    let local_dims = get_tensor_dims_iter(tensor_iter.clone());
    assert_eq!(local_dims.len(), 3);
    let mut dims = vec![0; local_dims.len()];
    comm.all_reduce_into(&local_dims, &mut dims, SystemOperation::max());

    // Organize into slices
    let mut slices: Vec<Vec<Vec<usize>>> = vec![vec![]; dims[0]];
    let mut nnz = 0;
    for (co, _) in tensor_iter.clone() {
        slices[co.co(0)].push((0..co.modes()).map(|i| co.co(i)).collect());
        nnz += 1;
    }
    println!("nnz: {}", nnz);
    println!("density: {:e}", (nnz as f64 / dims.iter().product::<usize>() as f64));

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
    let nnz_per_fiber_mean = nnz as f64 / (dims[0] * dims[2]) as f64;
    let nnz_per_fiber_std_dev = (square_nnz_per_fiber_sum as f64 / nnz as f64
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
