use mpi::traits::*;
use mpi::collective::SystemOperation;
use sparse_tensor::SparseTensor;
use sparse_tensor::synthetic::{TensorOptions, gentensor};

fn main() {
    let universe = mpi::initialize().expect("failed to initialize MPI");
    let world = universe.world();

    let args: Vec<String> = std::env::args().map(|s| s.to_string()).collect();
    let opts_fname = &args[1];
    let file_data = std::fs::read_to_string(opts_fname).expect("missing file data");
    let tensor_opts: TensorOptions = serde_json::from_str(&file_data).expect("failed to parse tensor options");
    let tensor = gentensor(tensor_opts, &world);

    // Check for empty slices
    for m in 0..3 {
        let local_max_dim = tensor.co[m].iter().max().expect("missing max value") + 1;
        let mut max_dim: usize = 0;
        world.all_reduce_into(&local_max_dim, &mut max_dim, SystemOperation::max());
        let mut local_nnz_counts = vec![0; max_dim];
        for co_val in &tensor.co[m] {
            local_nnz_counts[*co_val] += 1;
        }
        if world.rank() == 0 {
            let mut nnz_counts = vec![0; max_dim];
            world
                .process_at_rank(0)
                .reduce_into_root(
                    &local_nnz_counts,
                    &mut nnz_counts,
                    SystemOperation::sum(),
                );
            if nnz_counts.iter().any(|&count| count == 0) {
                eprintln!("WARNNIG: empty slices for mode {}", m);
            }
        } else {
            world
                .process_at_rank(0)
                .reduce_into(&local_nnz_counts, SystemOperation::sum());
        }
    }
}
