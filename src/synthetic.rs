//! Synthetic tensor generator.
//!
//! Work based on Torun et al. A Sparse Tensor Generator with Efficient Feature
//! Extraction. 2025.
use std::path::Path;
use std::io::prelude::*;
use std::io::BufWriter;
use std::collections::HashSet;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct TensorOptions {
    /// Tensor dimensions (3-way tensors only for now)
    dims: Vec<usize>,
    /// Density of the tensor
    nnz_density: f64,
    /// Fiber density
    fiber_density: f64,
    /// Coefficient of variation of fibers per slice
    cv_fibers_per_slice: f64,
    /// Coefficient of variation of nonzeros per fiber
    cv_nonzeros_per_fiber: f64,
    // TODO: Deal with imbalance
    // /// Imablance of fibers per slice
    // imbal_fiber_per_slice: f64,
    // /// Imbalance of nonzeros per slice
    // imbal_nonzeros_per_fiber: f64,
}

/// Return n indices each uniformly distributed from [0, limit)
fn randinds<R: Rng>(n: usize, limit: usize, rng: &mut R) -> Vec<usize> {
    assert!(limit > 0);
    let distr = Uniform::new(0, limit - 1).expect("failed to create uniform distribution");
    (0..n).map(|_| distr.sample(rng)).collect()
}

/// Implementation of the distribute function from the "A Sparse Tensor Generator
/// with Efficient Feature Extraction".
fn distribute<R: Rng>(
    n: usize,
    count_requested: usize,
    mean: f64,
    std_dev: f64,
    max: usize,
    limit: usize,
    rng: &mut R,
) -> (Vec<usize>, Vec<Vec<usize>>) {
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
    let distr = distr.expect("failed to create distribution");

    // Generate the counts
    let mut counts = vec![];
    for _ in 0..n {
        let count = if use_normal {
            // Use a normal distribution
            distr.sample(rng) as usize
        } else {
            // Use a log-normal distribution
            distr.sample(rng).exp() as usize
            // counts.push(distr.sample(rng).exp() as usize);
        };
        let count = if count > max { max } else { count };
        counts.push(count);
    }

    // Compare the computed mean with desired mean and scale if it doesn't match exactly
    let total: usize = counts.iter().sum();
    let ratio = count_requested as f64 / total as f64;
    if ratio < 0.95 || ratio > 1.05 {
        for count in &mut counts {
            *count = ((*count as f64) * ratio) as usize;
        }
    }

    // Now generate random indices
    let mut inds = vec![];
    let mut total = 0;
    for count in counts.iter_mut() {
        *count = std::cmp::min(*count, max);
        // I've noticed that setting the count to 1, if it is zero, can triple
        // the number of nonzeros, or worse. So, commenting it out.
        // *count = std::cmp::max(*count, 1);
        total += *count;
        // Create an array of size counts[i] all in the range [1, limit] ---
        // this is done with a uniform distribution here
        inds.push(randinds(*count, limit, rng));
    }

    (counts, inds)
}

/// Generate a tensor based on the input metrics.
///
/// Based on the following paper:
///
/// Torun et al. A Sparse Tensor Generator with Efficient Feature Extraction. 2025.
fn gentensor<P: AsRef<Path>>(tensor_fname: P, tensor_opts: TensorOptions) {
    let nnz = (tensor_opts.nnz_density * (tensor_opts.dims[0] * tensor_opts.dims[1]
                                          * tensor_opts.dims[2]) as f64) as usize;
    let slice_count = tensor_opts.dims[0];
    let nonzero_fiber_count = (tensor_opts.fiber_density
                               * (slice_count * tensor_opts.dims[1]) as f64) as usize;
    let mean_fibers_per_slice = nonzero_fiber_count as f64 / slice_count as f64;
    let std_dev_fibers_per_slice = tensor_opts.cv_fibers_per_slice * mean_fibers_per_slice;
    let max_fibers_per_slice = tensor_opts.dims[1];

    // Choose random indices for the slices
    // TODO: We need a deterministic and portable RNG here (see
    // https://rust-random.github.io/book/crate-reprod.html#crate-versions).
    // It looks like ChaCha20Rng could be useful (see
    // https://rust-random.github.io/book/guide-seeding.html#the-seed-type)
    let mut rng = rand::rng();
    // Distribute the number and indices of fibers per slice
    let (count_fibers_per_slice, fiber_indices_per_slice) = distribute(
        slice_count,
        nonzero_fiber_count,
        mean_fibers_per_slice,
        std_dev_fibers_per_slice,
        max_fibers_per_slice,
        tensor_opts.dims[1],
        &mut rng
    );
    let true_nonzero_fiber_count: usize = count_fibers_per_slice.iter().sum();

    // Compute nonzeros per fiber
    let mean_nonzeros_per_fiber = nnz as f64 / nonzero_fiber_count as f64;
    let std_dev_nonzeros_per_fiber = tensor_opts.cv_nonzeros_per_fiber * mean_nonzeros_per_fiber;
    let max_nonzeros_per_fiber = tensor_opts.dims[2];
    let (count_nonzeros_per_fiber, nonzero_indices_per_fiber) = distribute(
        true_nonzero_fiber_count,
        nnz,
        mean_nonzeros_per_fiber,
        std_dev_nonzeros_per_fiber,
        max_nonzeros_per_fiber,
        tensor_opts.dims[2],
        &mut rng,
    );
    let f = std::fs::File::create(tensor_fname).expect("failed to create file");
    let mut tensor_file = BufWriter::new(f);
    let value_distr = Uniform::new(0.0, 1.0).expect("failed to create uniform distribution for tensor values");
    // Iterate over all slices
    let mut fiber_idx = 0;
    let mut total_nnz = 0;
    for i in 0..slice_count {
        let mut slice_coords = HashSet::new();
        // Iterate over all fibers of this slice
        for j in 0..count_fibers_per_slice[i] {
            for k in 0..count_nonzeros_per_fiber[fiber_idx] {
                let value: f64 = value_distr.sample(&mut rng);
                let co = (fiber_indices_per_slice[i][j], nonzero_indices_per_fiber[fiber_idx][k]);
                // Skip duplicate coordinates
                //if slice_coords.contains(&co) {
                //    continue;
                //}
                writeln!(&mut tensor_file, "{} {} {} {:.4}", i + 1, co.0 + 1, co.1 + 1, value)
                    .expect("failed to write tensor entry");
                slice_coords.insert(co);
            }
            fiber_idx += 1;
        }
        total_nnz += slice_coords.len();
    }
}

use std::os::raw::{c_char, c_void};
use std::ffi::CStr;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sparse_tensor_synthetic_options_load(fname: *const c_char) -> *mut c_void {
    let fname = CStr::from_ptr(fname).to_string_lossy().to_string();
    let file_text_result = std::fs::read_to_string(&fname);
    if file_text_result.is_err() {
        eprintln!("sparse_tensor: failed to open {}", fname);
        return std::ptr::null_mut();
    }
    let file_text = file_text_result.expect("should not be possible");
    let opts: TensorOptions = serde_json::from_str(&file_text).expect("failed to parse tensor options");
    Box::into_raw(Box::new(opts)) as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sparse_tensor_synthetic_options_free(opts_handle: *mut c_void) {
    Box::from_raw(opts_handle as *mut TensorOptions);
}

#[unsafe(no_mangle)]
pub extern "C" fn sparse_tensor_synthetic_generate(opts_handle: *mut c_void) {
    // TODO
}

/*
use clap::Parser;
/// Synthetic tensor generator tool
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File name or path to output tensor
    #[arg(short, long, required = true)]
    fname: String,

    /// Dimensions in form {num}x{num}x...
    #[arg(short, long, required = true)]
    dims: String,

    /// Density of the tensor
    #[arg(long, required = true)]
    density: f64,

    /// Density of fibers in the tensor
    #[arg(long, required = true)]
    fiber_density: f64,

    /// Coefficient of variation of number of fibers per slice
    #[arg(long, required = true)]
    cv_fiber_slice: f64,

    /// Coefficient of variation of number of nonzeros per fiber
    #[arg(long, required = true)]
    cv_nonzero_fiber: f64,
}

fn main() {
    let args = Args::parse();

    let dims: Vec<usize> = args.dims
        .split('x')
        .map(|v| usize::from_str_radix(v, 10).expect("invalid dimensions specified"))
        .collect();
    // Only supporting 3-way tensors for right now
    assert_eq!(dims.len(), 3);
    gentensor(args.fname, TensorOptions {
        dims,
        nnz_density: args.density,
        fiber_density: args.fiber_density,
        cv_fibers_per_slice: args.cv_fiber_slice,
        cv_nonzeros_per_fiber: args.cv_nonzero_fiber,
    });
}
*/
