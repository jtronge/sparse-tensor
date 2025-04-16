use std::collections::HashSet;
use mpi::traits::*;
use sparse_tensor::synthetic::{TensorOptions, gentensor};

fn main() {
    let universe = mpi::initialize().expect("failed to initialize MPI universe");
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    println!("rank {} of {}", rank, size);
    let opts_file_data = std::fs::read_to_string("data/tensor_opts.json")
        .expect("failed to read in tensor opts file");
    let opts: TensorOptions = serde_json::from_str(&opts_file_data)
        .expect("failed to parse tensor options");
    let (co, vals) = gentensor(opts, &world);
    assert_eq!(co[0].len(), vals.len());
    let nnz = vals.len();

    // Check that every nonzero is unique
    let mut co_set = HashSet::new();
    let mut nonunique_count = 0;
    for i in 0..nnz {
        let ids: Vec<usize> = (0..3).map(|j| co[j][i]).collect();
        if !co_set.contains(&ids[..]) {
            co_set.insert(ids);
        }
    }
    let duplicate_percent = (nnz - co_set.len()) as f64 / nnz as f64;
    println!("{} duplicates", duplicate_percent);
}
