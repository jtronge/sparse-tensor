//! Synthetic tensor generator.
//!
//! Work based on Torun et al. A Sparse Tensor Generator with Efficient Feature
//! Extraction. 2025.
use std::collections::HashSet;
use std::os::raw::c_int;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, Deserialize};

use crate::comm::{self, Comm};

mod c_api;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TensorOptions {
    /// Tensor dimensions (3-way tensors only for now)
    pub dims: Vec<usize>,
    /// Density of the tensor
    pub nnz_density: f64,
    /// Fiber density
    pub fiber_density: f64,
    /// Coefficient of variation of fibers per slice
    pub cv_fibers_per_slice: f64,
    /// Coefficient of variation of nonzeros per fiber
    pub cv_nonzeros_per_fiber: f64,
    // TODO: Deal with imbalance
    // /// Imablance of fibers per slice
    // imbal_fiber_per_slice: f64,
    // /// Imbalance of nonzeros per slice
    // imbal_nonzeros_per_fiber: f64,
    /// Seed for the RNG.
    pub seed: u64,
}

/// Return n indices each uniformly distributed from [0, limit)
fn randinds<R: Rng>(n: usize, limit: usize, rng: &mut R) -> Vec<usize> {
    assert!(limit > 0);
    let distr = Uniform::new(0, limit - 1).expect("failed to create uniform distribution");
    (0..n).map(|_| distr.sample(rng)).collect()
}

/// Random distribution for counts (number of slices, fibers, nonzeros, etc.).
struct CountDistribution {
    /// Distribution handle.
    distr: Normal<f64>,

    /// Use the normal distribution or the log-normal distribution by default
    use_normal: bool,

    /// Maximum value allowed.
    max: usize,
}

impl CountDistribution {
    /// Create a distribution for a count value.
    fn new(mean: f64, std_dev: f64, max: usize) -> CountDistribution {
        // Check whether the normal distribution could generate many negative values
        let use_normal = mean > (3.0 * std_dev);
        let distr = if use_normal {
            Normal::new(mean, std_dev)
        } else {
            // Use log-normal only if there is potential for a lot of negative values
            let mean_log_norm = (mean * mean / (mean * mean + std_dev * std_dev).sqrt()).ln();
            let std_dev_log_norm = (1.0 + std_dev * std_dev / (mean * mean)).ln().sqrt();
            Normal::new(mean_log_norm, std_dev_log_norm)
        };
        let distr = distr.expect("failed to create normal distribution");

        CountDistribution {
            distr,
            use_normal,
            max,
        }
    }

    /// Sample a random count value.
    fn sample_count<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let count = if self.use_normal {
            // Use a normal distribution
            self.distr.sample(rng) as usize
        } else {
            // Use a log-normal distribution
            self.distr.sample(rng).exp() as usize
        };
        let count = if count > self.max { self.max } else { count };
        count as usize
    }
}

/// Container RNGs assigned to each slice of the tensor.
struct SliceRng {
    /// Start slice on this process
    local_start_slice: usize,
    /// RNGs for each slice
    rngs: Vec<ChaCha8Rng>,
}

impl SliceRng {
    fn new(tensor_opts: &TensorOptions, local_start_slice: usize, local_nslices: usize) -> SliceRng {
        let rngs: Vec<ChaCha8Rng> = (local_start_slice..local_start_slice+local_nslices)
            .map(|slice| ChaCha8Rng::seed_from_u64(tensor_opts.seed + slice as u64))
            .collect();
        SliceRng {
            local_start_slice,
            rngs,
        }
    }

    /// Get an RNG for a specific slice.
    pub fn rng(&mut self, slice: usize) -> &mut ChaCha8Rng {
        &mut self.rngs[slice - self.local_start_slice]
    }
}

/// Compare the computed mean with desired mean and scale if it doesn't match exactly.
///
/// This performs an allreduce to get the total counts across all ranks.
fn global_compare_with_expected_and_scale(total_requested: usize, counts: &mut [usize], comm: &Comm) {
    let local_count: u64 = counts.iter().sum::<usize>() as u64;
    // Do an allreduce to get the global total.
    let mut global_count: u64 = 0;
    comm.allreduce(&local_count, &mut global_count, comm::Operation::Sum);
    let ratio = total_requested as f64 / global_count as f64;
    // Scale the counts if the total sum is too large or too small.
    if ratio < 0.95 || ratio > 1.05 {
        for count in counts {
            *count = ((*count as f64) * ratio) as usize;
        }
    }
}

/// Distribute fibers per slice, in parallel, with a different RNG used for each slice.
///
/// Based on the distribute function from the "A Sparse Tensor Generator
/// with Efficient Feature Extraction".
fn distribute_fibers_per_slice(
    local_start_slice: usize,
    local_nslices: usize,
    total_fibers_requested: usize,
    mean: f64,
    std_dev: f64,
    max: usize,
    limit: usize,
    comm: &Comm,
    slice_rng: &mut SliceRng,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let count_distr = CountDistribution::new(mean, std_dev, max);

    // Generate the fiber counts per slice
    let mut fcounts_per_slice = vec![];
    for slice in local_start_slice..local_start_slice + local_nslices {
        fcounts_per_slice.push(count_distr.sample_count(slice_rng.rng(slice)));
    }

    global_compare_with_expected_and_scale(total_fibers_requested, &mut fcounts_per_slice[..], comm);

    // Now generate random indices for the fibers
    let mut fiber_inds = vec![];
    for (slice, fiber_count) in (local_start_slice..local_start_slice + local_nslices).zip(fcounts_per_slice.iter_mut()) {
        *fiber_count = std::cmp::min(*fiber_count, max);
        // I've noticed that setting the count to 1, if it is zero, can triple
        // the number of nonzeros, or worse. So, commenting it out.
        // *count = std::cmp::max(*count, 1);

        // Create an array of size fcounts_per_slice[slice] all in the range [1, limit] ---
        // this is done with a uniform distribution here
        fiber_inds.push(randinds(*fiber_count, limit, slice_rng.rng(slice)));
    }

    (fcounts_per_slice, fiber_inds)
}

/// Distribute nonzero indices per fiber.
///
/// This also does an allreduce to ensure that the counts matches the desired amount.
fn distribute_nnzs_per_fiber(
    local_start_slice: usize,
    local_nslices: usize,
    count_fibers_per_slice: &[usize],
    local_nonzero_fiber_count: usize,
    requested_nnz_count: usize,
    mean: f64,
    std_dev: f64,
    max: usize,
    limit: usize,
    comm: &Comm,
    slice_rng: &mut SliceRng,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let count_distr = CountDistribution::new(mean, std_dev, max);

    // Generate the counts
    let mut nnz_counts = vec![];
    for slice in local_start_slice..local_start_slice + local_nslices {
        let fiber_count = count_fibers_per_slice[slice - local_start_slice];
        for _ in 0..fiber_count {
            nnz_counts.push(count_distr.sample_count(slice_rng.rng(slice)));
        }
    }
    assert_eq!(nnz_counts.len(), local_nonzero_fiber_count);

    // Compare the computed mean with desired mean and scale if it doesn't match exactly
    global_compare_with_expected_and_scale(requested_nnz_count, &mut nnz_counts[..], comm);

    // Now generate random indices
    let mut nnz_inds = vec![];
    let mut last_count_idx = 0;
    for slice in local_start_slice..local_start_slice + local_nslices {
        let fiber_count = count_fibers_per_slice[slice - local_start_slice];
        for fiber in 0..fiber_count {
            let nnz_count = &mut nnz_counts[last_count_idx + fiber];
            *nnz_count = std::cmp::min(*nnz_count, max);
            // I've noticed that setting the count to 1, if it is zero, can triple
            // the number of nonzeros, or worse. So, commenting it out.
            // *count = std::cmp::max(*count, 1);

            // Create an array of size counts[i] all in the range [1, limit] ---
            // this is done with a uniform distribution here
            nnz_inds.push(randinds(*nnz_count, limit, slice_rng.rng(slice)));
        }
        last_count_idx += fiber_count;
    }

    (nnz_counts, nnz_inds)
}

/// Generate a tensor based on the input metrics.
///
/// Based on the following paper:
///
/// Torun et al. A Sparse Tensor Generator with Efficient Feature Extraction. 2025.
pub fn gentensor(tensor_opts: TensorOptions, comm: &Comm) -> (Vec<Vec<usize>>, Vec<f64>) {
    let size = comm.size();
    let rank = comm.rank();

    let slice_count = tensor_opts.dims[0];
    let slices_per_rank = slice_count / size;
    let local_start_slice = rank * slices_per_rank;
    // Each rank gets slices_per_rank slices, with the last one getting any leftovers
    let local_nslices = slices_per_rank + if rank == (size - 1) && size != 1 { slice_count % rank } else { 0 };
    assert!(local_nslices > 0);
    let mut slice_rng = SliceRng::new(&tensor_opts, local_start_slice, local_nslices);

    let nnz = (tensor_opts.nnz_density * (tensor_opts.dims[0] * tensor_opts.dims[1]
                                          * tensor_opts.dims[2]) as f64) as usize;
    let slice_count = tensor_opts.dims[0];
    let nonzero_fiber_count = (tensor_opts.fiber_density
                               * (slice_count * tensor_opts.dims[1]) as f64) as usize;
    let mean_fibers_per_slice = nonzero_fiber_count as f64 / slice_count as f64;
    let std_dev_fibers_per_slice = tensor_opts.cv_fibers_per_slice * mean_fibers_per_slice;
    let max_fibers_per_slice = tensor_opts.dims[1];

    // Compute number and indicies of fibers per slice
    let (count_fibers_per_slice, fiber_indices) = distribute_fibers_per_slice(
        local_start_slice,
        local_nslices,
        nonzero_fiber_count,
        mean_fibers_per_slice,
        std_dev_fibers_per_slice,
        max_fibers_per_slice,
        tensor_opts.dims[1],
        comm,
        &mut slice_rng,
    );
    let local_nonzero_fiber_count: usize = count_fibers_per_slice.iter().sum();

    // Compute nonzeros per fiber
    let mean_nonzeros_per_fiber = nnz as f64 / nonzero_fiber_count as f64;
    let std_dev_nonzeros_per_fiber = tensor_opts.cv_nonzeros_per_fiber * mean_nonzeros_per_fiber;
    let max_nonzeros_per_fiber = tensor_opts.dims[2];
    let (count_nonzeros_per_fiber, nonzero_indices) = distribute_nnzs_per_fiber(
        local_start_slice,
        local_nslices,
        &count_fibers_per_slice[..],
        local_nonzero_fiber_count,
        nnz,
        mean_nonzeros_per_fiber,
        std_dev_nonzeros_per_fiber,
        max_nonzeros_per_fiber,
        tensor_opts.dims[2],
        comm,
        &mut slice_rng,
    );

    let value_distr = Uniform::new(0.0, 1.0).expect("failed to create uniform distribution for tensor values");
    let mut fiber_nnz_idx = 0;
    let mut co = vec![vec![]; 3];
    let mut vals = vec![];
    for slice in local_start_slice..local_start_slice + local_nslices {
        for fiber in 0..count_fibers_per_slice[slice - local_start_slice] {
            for k in 0..count_nonzeros_per_fiber[fiber_nnz_idx] {
                co[0].push(slice);
                co[1].push(fiber_indices[slice - local_start_slice][fiber]);
                co[2].push(nonzero_indices[fiber_nnz_idx][k]);
                vals.push(value_distr.sample(slice_rng.rng(slice)));
            }
            fiber_nnz_idx += 1;
        }
    }
    (co, vals)
}
