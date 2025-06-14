//! Synthetic tensor generator.
//!
//! Work based on Torun et al. A Sparse Tensor Generator with Efficient Feature
//! Extraction. 2025.
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, Deserialize};
use mpi::traits::*;
use mpi::collective::SystemOperation;
use crate::SparseTensor;

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
fn randinds<R: Rng>(
    n: usize,
    limit: usize,
    index_check: &mut [bool],
    rng: &mut R,
) -> Vec<usize> {
    assert_eq!(index_check.len(), limit);
    // TODO: This is probably what is taking the majority of the time
    // Should switch to a hashset at a certain limit or so
    index_check.fill(false);
    let distr = Uniform::new(0, limit - 1).expect("failed to create uniform distribution");
    let mut inds = vec![];
    for _ in 0..n {
        let i = distr.sample(rng);
        if index_check[i] {
            continue;
        }
        index_check[i] = true;
        inds.push(i);
    }
    inds
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
fn global_compare_with_expected_and_scale<C>(total_requested: usize, counts: &mut [usize], comm: &C)
where
    C: AnyCommunicator,
{
    let local_count: u64 = counts.iter().sum::<usize>() as u64;
    // Do an allreduce to get the global total.
    let mut global_count: u64 = 0;
    comm.all_reduce_into(&local_count, &mut global_count, SystemOperation::sum());
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
fn distribute_fibers_per_slice<C>(
    local_start_slice: usize,
    local_nslices: usize,
    total_fibers_requested: usize,
    mean: f64,
    std_dev: f64,
    max: usize,
    limit: usize,
    comm: &C,
    slice_rng: &mut SliceRng,
) -> (Vec<usize>, Vec<Vec<usize>>)
where
    C: AnyCommunicator,
{
    let count_distr = CountDistribution::new(mean, std_dev, max);

    // Generate the fiber counts per slice
    let mut fcounts_per_slice = vec![];
    for slice in local_start_slice..local_start_slice + local_nslices {
        fcounts_per_slice.push(count_distr.sample_count(slice_rng.rng(slice)));
    }

    global_compare_with_expected_and_scale(total_fibers_requested, &mut fcounts_per_slice[..], comm);

    // Now generate random indices for the fibers
    let mut fiber_inds = vec![];
    let mut index_check = vec![false; limit];
    for (slice, fiber_count) in (local_start_slice..local_start_slice + local_nslices).zip(fcounts_per_slice.iter_mut()) {
        *fiber_count = std::cmp::min(*fiber_count, max);
        // I've noticed that setting the count to 1, if it is zero, can triple
        // the number of nonzeros, or worse. So, commenting it out.
        // *count = std::cmp::max(*count, 1);

        // Create an array of size fcounts_per_slice[slice] all in the range [1, limit] ---
        // this is done with a uniform distribution here
        let inds = randinds(*fiber_count, limit, &mut index_check[..], slice_rng.rng(slice));
        *fiber_count = inds.len();
        fiber_inds.push(inds);
    }

    (fcounts_per_slice, fiber_inds)
}

/// Distribute nonzero indices per fiber.
///
/// This also does an allreduce to ensure that the counts matches the desired amount.
fn distribute_nnzs_per_fiber<C>(
    local_start_slice: usize,
    local_nslices: usize,
    count_fibers_per_slice: &[usize],
    local_nonzero_fiber_count: usize,
    requested_nnz_count: usize,
    mean: f64,
    std_dev: f64,
    max: usize,
    limit: usize,
    comm: &C,
    slice_rng: &mut SliceRng,
) -> (Vec<usize>, Vec<Vec<usize>>)
where
    C: AnyCommunicator,
{
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
    let mut index_check = vec![false; limit];
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
            let inds = randinds(*nnz_count, limit, &mut index_check[..], slice_rng.rng(slice));
            *nnz_count = inds.len();
            nnz_inds.push(inds);
        }
        last_count_idx += fiber_count;
    }

    (nnz_counts, nnz_inds)
}

/// Remove empty slices from the tensor coordinates.
fn remove_empty_slices<C: AnyCommunicator>(co: &mut [Vec<usize>], comm: &C) {
    let max_dims: Vec<usize> = co
        .iter()
        .map(|co_vals| co_vals.iter().max().unwrap_or(&0) + 1)
        .collect();
    let mut global_max_dims: Vec<usize> = vec![0; co.len()];
    comm.all_reduce_into(
        &max_dims[..],
        &mut global_max_dims[..],
        SystemOperation::max(),
    );

    // Compute the nonzero slices
    let mut nonzero_slices: Vec<Vec<u8>> = (0..co.len())
        .map(|m| vec![0; global_max_dims[m]])
        .collect();
    for i in 0..co[0].len() {
        for m in 0..co.len() {
            nonzero_slices[m][co[m][i]] = 1;
        }
    }

    let mut index_shifts: Vec<Vec<usize>> = vec![];
    for m in 0..co.len() {
        // Compute the global nonzero slices.
        let mut global_nonzero_slices: Vec<u8> = vec![0; global_max_dims[m]];
        comm
            .all_reduce_into(
                &nonzero_slices[m],
                &mut global_nonzero_slices,
                SystemOperation::sum(),
            );
        // Compute the amount to shift every index over by to remove the empty
        // slices.
        let shifts: Vec<usize> = global_nonzero_slices
            .iter()
            .map(|&nonzero| if nonzero == 0 { 1 } else { 0 })
            .scan(0, |state, count| {
                *state += count;
                Some(*state)
            })
            .collect();
        index_shifts.push(shifts);
    }

    // Remove the empty slices
    for i in 0..co[0].len() {
        for m in 0..co.len() {
            co[m][i] -= index_shifts[m][co[m][i]];
        }
    }
}

struct TensorGenerator<'a, C> {
    /// Input tensor options to generate for.
    opts: TensorOptions,

    /// Communicator object.
    comm: &'a C,

    /// Total number of expected nonzeros.
    nnz: usize,

    /// Start slice on this rank.
    local_start_slice: usize,

    /// Number of slices on this rank.
    local_nslices: usize,

    slice_rng: SliceRng,

    nonzero_fiber_count: usize,

    mean_fibers_per_slice: f64,

    std_dev_fibers_per_slice: f64,

    max_fibers_per_slice: usize,

    local_nonzero_fiber_count: Option<usize>,

    count_fibers_per_slice: Option<Vec<usize>>,

    fiber_indices: Option<Vec<Vec<usize>>>,

    count_nonzeros_per_fiber: Option<Vec<usize>>,

    nonzero_indices: Option<Vec<Vec<usize>>>,

    /// Generated tensor coordinates.
    pub co: Option<Vec<Vec<usize>>>,

    /// Generated tensor values.
    pub vals: Option<Vec<f64>>,
}

impl<'a, C> TensorGenerator<'a, C>
where
    C: AnyCommunicator
{
    pub fn new(tensor_opts: TensorOptions, comm: &'a C) -> TensorGenerator<'a, C> {
        let size: usize = comm.size().try_into().expect("failed to convert size");
        let rank: usize = comm.rank().try_into().expect("failed to convert size");

        let slice_count = tensor_opts.dims[0];
        let slices_per_rank = slice_count / size;
        let local_start_slice = rank * slices_per_rank;
        // Each rank gets slices_per_rank slices, with the last one getting any leftovers
        let local_nslices = slices_per_rank + if rank == (size - 1) && size != 1 { slice_count % rank } else { 0 };
        let slice_rng = SliceRng::new(&tensor_opts, local_start_slice, local_nslices);
        let nnz = (tensor_opts.nnz_density * (tensor_opts.dims[0] * tensor_opts.dims[1]
                                          * tensor_opts.dims[2]) as f64) as usize;
        let slice_count = tensor_opts.dims[0];
        let nonzero_fiber_count = (tensor_opts.fiber_density
                                   * (slice_count * tensor_opts.dims[1]) as f64) as usize;
        let mean_fibers_per_slice = nonzero_fiber_count as f64 / slice_count as f64;
        let std_dev_fibers_per_slice = tensor_opts.cv_fibers_per_slice * mean_fibers_per_slice;
        let max_fibers_per_slice = tensor_opts.dims[1];

        TensorGenerator {
            opts: tensor_opts,
            comm,
            nnz,
            local_start_slice,
            local_nslices,
            slice_rng,
            nonzero_fiber_count,
            mean_fibers_per_slice,
            std_dev_fibers_per_slice,
            max_fibers_per_slice,
            local_nonzero_fiber_count: None,
            count_fibers_per_slice: None,
            fiber_indices: None,
            count_nonzeros_per_fiber: None,
            nonzero_indices: None,
            co: None,
            vals: None,
        }
    }

    /// Compute number and indicies of fibers per slice
    fn compute_fibers(&mut self) {
        let (count_fibers_per_slice, fiber_indices) = distribute_fibers_per_slice(
            self.local_start_slice,
            self.local_nslices,
            self.nonzero_fiber_count,
            self.mean_fibers_per_slice,
            self.std_dev_fibers_per_slice,
            self.max_fibers_per_slice,
            self.opts.dims[1],
            self.comm,
            &mut self.slice_rng,
        );
        let local_nonzero_fiber_count: usize = count_fibers_per_slice.iter().sum();
        let _ = self.local_nonzero_fiber_count.insert(local_nonzero_fiber_count);
        let _ = self.count_fibers_per_slice.insert(count_fibers_per_slice);
        let _ = self.fiber_indices.insert(fiber_indices);
    }

    /// Compute nonzeros per fiber
    fn compute_nnzs(&mut self) {
        let mean_nonzeros_per_fiber = self.nnz as f64 / self.nonzero_fiber_count as f64;
        let std_dev_nonzeros_per_fiber = self.opts.cv_nonzeros_per_fiber * mean_nonzeros_per_fiber;
        let max_nonzeros_per_fiber = self.opts.dims[2];
        let (count_nonzeros_per_fiber, nonzero_indices) = distribute_nnzs_per_fiber(
            self.local_start_slice,
            self.local_nslices,
            &self.count_fibers_per_slice
                .as_ref()
                .expect("missing count fibers per slice"),
            self.local_nonzero_fiber_count
                .expect("missing local nonzero fiber count"),
            self.nnz,
            mean_nonzeros_per_fiber,
            std_dev_nonzeros_per_fiber,
            max_nonzeros_per_fiber,
            self.opts.dims[2],
            self.comm,
            &mut self.slice_rng,
        );
        let _ = self.count_nonzeros_per_fiber.insert(count_nonzeros_per_fiber);
        let _ = self.nonzero_indices.insert(nonzero_indices);
    }

    /// Fill entries of the tensor.
    fn fill_entries(&mut self) {
        // Value distribution for entry values.
        let value_distr = Uniform::new(0.0, 1.0)
            .expect("failed to create uniform distribution for tensor values");
        let mut fiber_nnz_idx = 0;
        let mut co = vec![vec![]; 3];
        let mut vals = vec![];
        let count_fps = self.count_fibers_per_slice
            .as_ref()
            .expect("missing count fibers per slice");
        let fiber_indices = self.fiber_indices
            .as_ref()
            .expect("missing fiber indices");
        let count_npf = self.count_nonzeros_per_fiber
            .as_ref()
            .expect("missing count nonzeros per fiber");
        let nonzero_indices = self.nonzero_indices
            .as_ref()
            .expect("missing nonzero indices");
        let start_slice = self.local_start_slice;
        let limit_slice = self.local_start_slice + self.local_nslices;
        for slice in start_slice..limit_slice {
            for fiber in 0..count_fps[slice - self.local_start_slice] {
                let fiber_idx = fiber_indices[slice - self.local_start_slice][fiber];
                for k in 0..count_npf[fiber_nnz_idx] {
                    // Add the nonzero
                    co[0].push(slice);
                    co[1].push(fiber_idx);
                    co[2].push(nonzero_indices[fiber_nnz_idx][k]);
                    vals.push(value_distr.sample(self.slice_rng.rng(slice)));
                }
                fiber_nnz_idx += 1;
            }
        }
        let _ = self.co.insert(co);
        let _ = self.vals.insert(vals);
    }
}

/// Generate a tensor based on the input metrics.
///
/// Based on the following paper:
///
/// Torun et al. A Sparse Tensor Generator with Efficient Feature Extraction. 2025.
pub fn gentensor<C>(tensor_opts: TensorOptions, comm: &C) -> SparseTensor
where
    C: AnyCommunicator
{
    let mut generator = TensorGenerator::new(tensor_opts, comm);
    generator.compute_fibers();
    generator.compute_nnzs();
    generator.fill_entries();
/*

    // Check for and remove any empty slices
    let empty_slice_timer = Instant::now();
    remove_empty_slices(&mut co[..], comm);
    if rank == 0 {
        println!("==> remove_empty_slice_time={}s",
                 empty_slice_timer.elapsed().as_secs_f64());
    }

    if rank == 0 {
        println!("==> sparse_tensor_generate_time={}s", now.elapsed().as_secs_f64());
    }
*/
    SparseTensor::new(
        generator.vals.take().expect("missing values"),
        generator.co.take().expect("missing coordinates"),
    )
}
