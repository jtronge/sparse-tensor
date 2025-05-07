//! Tensor feature analysis and helping library.
use std::time::Instant;
use mpi::Count;
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::*;
use mpi::collective::SystemOperation;
use crate::{SparseTensor, format_dims};

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

/// Determine the owner of a slice.
fn determine_owner(first_slices: &[usize], slice: usize) -> usize {
    for i in 0..first_slices.len() {
        if slice >= first_slices[i] && slice < first_slices[i+1] {
            return i;
        }
    }
    // Just return the last rank
    return first_slices.len() - 1;
}

/// Redistribute the input tensor across ranks using a 1D distribution.
fn redistribute_1d<C>(
    tensor: &SparseTensor,
    dims: &[usize],
    distribute_mode: usize,
    comm: &C,
) -> SparseTensor
where
    C: AnyCommunicator,
{
    // Determine number of slices per rank in the 1D decomposition
    let size: usize = comm.size().try_into().expect("failed to convert communicator size to usize");
    let slice_count = dims[distribute_mode];
    let slices_per_rank = slice_count / size;
    let mut first_slices: Vec<usize> = (0..size).map(|rank| {
        let count = rank * slices_per_rank;
        if rank == (size - 1) && size > 1 {
            // Put the remainder on the last rank
            count + slice_count % rank
        } else {
            count
        }
    }).collect();
    first_slices.push(slice_count);

    // Go through my local nonzeros and determine where they need to go.
    let mut owner_to_index = vec![vec![]; size];
    let mut send_counts: Vec<Count> = vec![0; size];
    for i in 0..tensor.count() {
        let owner = determine_owner(&first_slices[..], tensor.co[distribute_mode][i]);
        owner_to_index[owner].push(i);
        send_counts[owner] += 1;
    }
    let mut nnzs_to_send = vec![vec![]; tensor.modes()];
    let mut values_to_send = vec![];
    let mut send_disps: Vec<Count> = vec![0];
    for rank in 0..size {
        for m in 0..tensor.modes() {
            for i in &owner_to_index[rank] {
                nnzs_to_send[m].push(tensor.co[m][*i]);
            }
        }
        for i in &owner_to_index[rank] {
            values_to_send.push(tensor.values[*i]);
        }
        if rank > 0 {
            send_disps.push(send_disps[rank - 1] + send_counts[rank - 1]);
        }
    }
    let mut recv_counts: Vec<Count> = vec![0; size];
    comm.all_to_all_into(&send_counts[..], &mut recv_counts[..]);
    let mut recv_disps: Vec<Count> = vec![0];
    for rank in 1..size {
        recv_disps.push(recv_disps[rank - 1] + recv_counts[rank - 1]);
    }
    let total_recv: usize = (recv_disps[recv_disps.len() - 1]
                             + recv_counts[recv_counts.len() - 1])
        .try_into().expect("failed to convert i32 to usize");

    let mut recv_co = vec![];
    for m in 0..tensor.modes() {
        let send_buffer = Partition::new(&nnzs_to_send[m][..], &send_counts[..], &send_disps[..]);
        let mut tmp_buf = vec![0; total_recv];
        let mut recv_buffer = PartitionMut::new(&mut tmp_buf[..], &recv_counts[..], &recv_disps[..]);
        comm.all_to_all_varcount_into(&send_buffer, &mut recv_buffer);
        recv_co.push(tmp_buf);
    }

    let mut recv_vals = vec![0.0; total_recv];
    let send_buffer = Partition::new(&values_to_send[..], &send_counts[..], &send_disps[..]);
    let mut recv_buffer = PartitionMut::new(&mut recv_vals[..], &recv_counts[..], &recv_disps[..]);
    comm.all_to_all_varcount_into(&send_buffer, &mut recv_buffer);

    SparseTensor::new(recv_vals, recv_co)
}

/// Sort the tensor entries by mode and compute the slice pointers.
fn sort_compute_slice_ptrs(local_tensor: &mut SparseTensor) -> Vec<usize> {
    local_tensor.sort_by_mode(0);
    let mut slice_ptrs = vec![0];
    if local_tensor.count() > 0 {
        let mut last_slice = local_tensor.co[0][0];
        for i in 1..local_tensor.count() {
            let mut j = last_slice;
            while j != local_tensor.co[0][i] {
                slice_ptrs.push(i);
                j += 1;
            }
            last_slice = local_tensor.co[0][i];
        }
    }
    slice_ptrs.push(local_tensor.count());
    slice_ptrs
}

/// Abstraction for analyzing a tensor.
struct Analysis {
    /// The sparse tensor data (local only).
    local_tensor: SparseTensor,

    /// Global dimensions of the tensor.
    global_dims: Vec<usize>,

    /// Global number of nonzeros.
    global_nnz: usize,

    /// Local slice pointers (as with a CSF).
    local_slice_ptrs: Vec<usize>,
}

impl Analysis {
    fn new<C>(mut local_tensor: SparseTensor, global_dims: Vec<usize>, global_nnz: usize, comm: &C) -> Analysis
    where
        C: AnyCommunicator,
    {
        let sort_slice_timer = Instant::now();
        let local_slice_ptrs = sort_compute_slice_ptrs(&mut local_tensor);
        if comm.rank() == 0 {
            println!("==> sort_slice_time={}s", sort_slice_timer.elapsed().as_secs_f64());
        }

        Analysis {
            local_tensor,
            global_dims,
            global_nnz,
            local_slice_ptrs,
        }
    }

    /// Calculate properties of fibers per each slice globally.
    fn calculate_fibers_per_slice_props<C: AnyCommunicator>(&self, comm: &C) -> (f64, f64, f64, f64) {
        let mut fiber_count = 0;
        let mut fibers_per_slice_square = 0;
        let mut fibers_per_slice_max = 0;
        let mut nonzero_fiber_count_set = vec![0; self.global_dims[2]];
        // Iterate over each slice
        for i in 0..self.local_slice_ptrs.len()-1 {
            // These are the X(i, :, k) fibers
            // Count number of nonzeros in each fiber
            nonzero_fiber_count_set.fill(0);
            // Now iterate over every nonzero within the slice
            for j in self.local_slice_ptrs[i]..self.local_slice_ptrs[i+1] {
                let fiber_idx = self.local_tensor.co[2][j];
                nonzero_fiber_count_set[fiber_idx] += 1;
            }
            let nonzero_fiber_count = nonzero_fiber_count_set
                .iter()
                .map(|count| if *count > 0 { 1 } else { 0 })
                .sum();
            fiber_count += nonzero_fiber_count;
            fibers_per_slice_square += nonzero_fiber_count * nonzero_fiber_count;
            fibers_per_slice_max = std::cmp::max(
                fibers_per_slice_max,
                nonzero_fiber_count,
            );
        }

        // All-reduce communication.
        let mut tmp = [0usize; 2];
        comm.all_reduce_into(
            &[fiber_count, fibers_per_slice_square],
            &mut tmp,
            SystemOperation::sum(),
        );
        let global_fiber_count = tmp[0];
        let global_fibers_per_slice_square = tmp[1];
        let mut global_fibers_per_slice_max: usize = 0;
        comm.all_reduce_into(
            &fibers_per_slice_max,
            &mut global_fibers_per_slice_max,
            SystemOperation::max(),
        );

        let global_fibers_per_slice_density = global_fiber_count as f64
                                              / (self.global_dims[0]
                                                 * self.global_dims[2]) as f64;
        let global_fibers_per_slice_mean = global_fiber_count as f64
                                           / self.global_dims[0] as f64;
        let global_fibers_per_slice_std_dev = (global_fibers_per_slice_square as f64
                                               / self.global_dims[0] as f64
                                               - global_fibers_per_slice_mean
                                                 * global_fibers_per_slice_mean).sqrt();
        // Imbalance calculated as (max - average) / max
        let global_fibers_per_slice_imbalance = (global_fibers_per_slice_max as f64
                                                 - global_fibers_per_slice_mean)
                                                 / global_fibers_per_slice_max as f64;
        (
            global_fibers_per_slice_density,
            global_fibers_per_slice_mean,
            global_fibers_per_slice_std_dev,
            global_fibers_per_slice_imbalance,
        )
    }

    /// Calculate properties of nonzeros per each fiber globally.
    fn calculate_nnzs_per_fiber_props<C: AnyCommunicator>(&self, comm: &C) -> (f64, f64, f64) {
        let mut nnz_total = 0;
        let mut nnz_per_fiber_square = 0;
        let mut nnz_per_fiber_max = 0;
        // These are the X(i, :, k) fibers
        let mut nonzero_fibers_set = vec![0; self.global_dims[2]];
        // Iterate over each slice
        for i in 0..self.local_slice_ptrs.len()-1 {
            nonzero_fibers_set.fill(0);
            // Now iterate over every nonzero within the slice
            for j in self.local_slice_ptrs[i]..self.local_slice_ptrs[i+1] {
                let fiber_idx = self.local_tensor.co[2][j];
                nonzero_fibers_set[fiber_idx] += 1;
            }
            nnz_total += nonzero_fibers_set
                .iter()
                .sum::<usize>();
            nnz_per_fiber_square += nonzero_fibers_set
                .iter()
                .map(|&count| count * count)
                .sum::<usize>();
            let max_now = nonzero_fibers_set
                .iter()
                .max()
                .expect("failed to get max number of nnzs in a slice");
            nnz_per_fiber_max = std::cmp::max(nnz_per_fiber_max, *max_now);
        }

        // All-reduce communication.
        let mut tmp = [0usize; 2];
        comm.all_reduce_into(
            &[nnz_total, nnz_per_fiber_square],
            &mut tmp,
            SystemOperation::sum(),
        );
        let global_nnz = tmp[0];
        let global_nnz_per_fiber_square = tmp[1];
        assert_eq!(global_nnz, self.global_nnz);
        assert!(global_nnz_per_fiber_square > global_nnz);

        let mut global_nnz_per_fiber_max: usize = 0;
        comm.all_reduce_into(
            &nnz_per_fiber_max,
            &mut global_nnz_per_fiber_max,
            SystemOperation::max(),
        );

        let total_possible_fibers = (self.global_dims[0] * self.global_dims[2]) as f64;
        let global_nnz_per_fiber_mean = self.global_nnz as f64 / total_possible_fibers;
        let global_nnz_per_fiber_std_dev = (global_nnz_per_fiber_square as f64 / total_possible_fibers
                                            - global_nnz_per_fiber_mean * global_nnz_per_fiber_mean).sqrt();
        let global_nnz_per_fiber_imbalance = (global_nnz_per_fiber_max as f64 - global_nnz_per_fiber_mean)
                                             / global_nnz_per_fiber_max as f64;
        (
            global_nnz_per_fiber_mean,
            global_nnz_per_fiber_std_dev,
            global_nnz_per_fiber_imbalance,
        )
    }
}

pub fn analyze_tensor<C>(tensor: &SparseTensor, comm: &C)
where
    C: AnyCommunicator,
{
    let total_timer = Instant::now();
    let rank = comm.rank();

    // First determine the global dimensions of the tensor across all ranks.
    let dim_timer = Instant::now();
    let local_dims = tensor.tensor_dims();
    let mut dims = vec![0; local_dims.len()];
    comm.all_reduce_into(&local_dims, &mut dims, SystemOperation::max());
    if rank == 0 {
        println!("==> calculate_dimension_time={}s", dim_timer.elapsed().as_secs_f64());
    }

    // Redistribute the tensor using a 1D distribution across the ranks.
    let redistribute_timer = Instant::now();
    let local_tensor = redistribute_1d(tensor, &dims[..], 0, comm);
    if rank == 0 {
        println!("==> redistribute_time={}s", redistribute_timer.elapsed().as_secs_f64());
    }

    let nnz_timer = Instant::now();
    let local_nnz = local_tensor.count();
    let mut global_nnz = 0;
    comm.all_reduce_into(&local_nnz, &mut global_nnz, SystemOperation::sum());
    if rank == 0 {
        println!("==> compute_global_nnz_time={}s", nnz_timer.elapsed().as_secs_f64());
    }

    let analysis = Analysis::new(local_tensor, dims, global_nnz, comm);

    // Compute fibers per slice properties
    let fiber_timer = Instant::now();
    let (fps_density, fps_mean, fps_std_dev, fps_imbalance) = analysis.calculate_fibers_per_slice_props(comm);
    if rank == 0 {
        println!("==> compute_fibers_per_slice_time={}s", fiber_timer.elapsed().as_secs_f64())
    }

    // Compute nnzs per fiber properties
    let nnz_timer = Instant::now();
    let (npf_mean, npf_std_dev, npf_imbalance) = analysis.calculate_nnzs_per_fiber_props(comm);
    if rank == 0 {
        println!("==> compute_nnzs_per_fiber_time={}s", nnz_timer.elapsed().as_secs_f64());
    }

    if rank == 0 {
        println!("==> total_time={}s", total_timer.elapsed().as_secs_f64());
    }

    if rank == 0 {
        println!("dims: {}", format_dims(&analysis.global_dims[..]));
        println!("nnzs: {}", analysis.global_nnz);
        println!("density: {:e}", analysis.global_nnz as f64 / analysis.global_dims.iter().product::<usize>() as f64);
        println!("fiber_density: {:e}", fps_density);
        println!("fibers_per_slice_mean: {}", fps_mean);
        println!("fibers_per_slice_std_dev: {}", fps_std_dev);
        println!("fibers_per_slice_cv: {}", fps_std_dev / fps_mean);
        println!("fibers_per_slice_imbalance: {}", fps_imbalance);
        println!("nnz_per_fiber_mean: {}", npf_mean);
        println!("nnz_per_fiber_std_dev: {}", npf_std_dev);
        println!("nnz_per_fiber_cv: {}", npf_std_dev / npf_mean);
        println!("nnz_per_fiber_imbalance: {}", npf_imbalance);
    }
}
