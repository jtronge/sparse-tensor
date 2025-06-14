//! Test MPI sparse tensor generation and feature analysis.
use mpi::traits::*;
use mpi::topology::Color;
use mpi::datatype::PartitionMut;
use sparse_tensor::SparseTensor;
use sparse_tensor::synthetic::{TensorOptions, gentensor};

#[derive(Debug)]
struct MPITestError(String);

/// All gather a tensor across all ranks so that we can do comparisons and other
/// checks.
fn all_gather_tensor<C: AnyCommunicator>(local_tensor: SparseTensor, comm: &C) -> SparseTensor {
    // Get the counts
    let local_count: i32 = local_tensor.values
        .len()
        .try_into()
        .expect("failed to convert local tensor size");
    let comm_size = comm.size().try_into().expect("failed to convert communicator size");
    let mut counts = vec![0i32; comm_size];
    comm.all_gather_into(&local_count, &mut counts[..]);
    let total_count: usize = counts
        .iter()
        .sum::<i32>()
        .try_into()
        .expect("failed to convert total size");
    let mut disps = vec![0];
    for i in 1..comm_size {
        disps.push(disps[i-1] + counts[i-1]);
    }

    // All gather the coordinates
    let mut co = vec![];
    for i in 0..3 {
        let mut co_buf = vec![0usize; total_count];
        let mut co_part = PartitionMut::new(&mut co_buf[..], &counts[..], &disps[..]);
        comm.all_gather_varcount_into(&local_tensor.co[i], &mut co_part);
        co.push(co_buf);
    }
    // All gather the values
    let mut values = vec![0.0; total_count];
    let mut values_part = PartitionMut::new(&mut values[..], &counts[..], &disps[..]);
    comm.all_gather_varcount_into(&local_tensor.values, &mut values_part);

    SparseTensor {
        co,
        values,
    }
}

/// Assert macro return an error on failure of the predicate.
macro_rules! mpi_test_assert {
    ($predicate:expr) => {
        match &($predicate) {
            value => {
                if !*value {
                    return Err(
                        MPITestError(
                            format!("assertion failed: {}", stringify!($predicate))
                        )
                    );
                }
            }
        }
    };
}

/// Assert equal macro to return an error on comparison failure.
///
/// Based on built-in assert_eq macro.
macro_rules! mpi_test_assert_eq {
    ($left:expr, $right:expr) => {
        match (&$left, &$right) {
            (left, right) => {
                if !(*left == *right) {
                    return Err(
                        MPITestError(
                            format!("left {:?} is not equal to right {:?}", left, right)
                        )
                    );
                }
            }
        }
    };
}

fn basic_test<C: AnyCommunicator>(
    comm_large: &C,
    comm_medium: &C,
    comm_small: &C,
    comm_tiny: &C,
) -> Result<(), MPITestError> {
    let opts = TensorOptions {
        dims: vec![10, 10, 10],
        nnz_density: 0.1,
        fiber_density: 0.3,
        cv_fibers_per_slice: 1.0,
        cv_nonzeros_per_fiber: 1.0,
        // TODO: Imbalance
        seed: 100,
    };
    let tensor_large = all_gather_tensor(
        gentensor(opts.clone(), comm_large),
        comm_large,
    );
    let tensor_medium = all_gather_tensor(
        gentensor(opts.clone(), comm_medium),
        comm_medium,
    );
    let tensor_small = all_gather_tensor(
        gentensor(opts.clone(), comm_small),
        comm_small,
    );
    let tensor_tiny = all_gather_tensor(
        gentensor(opts.clone(), comm_tiny),
        comm_tiny,
    );

    mpi_test_assert!(tensor_large.values.len() > 0);
    mpi_test_assert_eq!(tensor_large.values, tensor_medium.values);
    mpi_test_assert_eq!(tensor_large.co, tensor_medium.co);
    mpi_test_assert_eq!(tensor_large.values, tensor_small.values);
    mpi_test_assert_eq!(tensor_large.co, tensor_small.co);
    mpi_test_assert_eq!(tensor_large.values, tensor_tiny.values);
    mpi_test_assert_eq!(tensor_large.co, tensor_tiny.co);
    Ok(())
}

fn main() {
    let universe = mpi::initialize().expect("failed to initialize MPI");
    let world = universe.world();
    assert_eq!(world.size(), 8);

    // Set up a communicator with size 8 and one with size 4
    let comm_large = world;
    let comm_medium = comm_large
        .split_by_color(Color::with_value(comm_large.rank() % 2))
        .expect("failed to split communicator");
    assert_eq!(comm_medium.size(), 4);
    let comm_small = comm_large
        .split_by_color(Color::with_value(comm_large.rank() % 4))
        .expect("failed to split communicator");
    assert_eq!(comm_small.size(), 2);
    let comm_tiny = comm_large
        .split_by_color(Color::with_value(comm_large.rank() % 8))
        .expect("failed to split communicator");
    assert_eq!(comm_tiny.size(), 1);

    basic_test(&comm_large, &comm_medium, &comm_small, &comm_tiny)
        .expect("failed basic test")

    // TODO
}
