use std::collections::HashSet;
use mpi::traits::*;
use mpi::datatype::PartitionMut;
use mpi::Count;
use std::collections::BTreeMap;
use sparse_tensor::synthetic::{TensorOptions, gentensor};
use sparse_tensor::feat;

fn main() {
    let universe = mpi::initialize().expect("failed to initialize MPI universe");
    let world = universe.world();
    let size: usize = world.size().try_into().expect("failed to convert size");
    let rank: usize = world.rank().try_into().expect("failed to convert rank");
    let opts_file_data = std::fs::read_to_string("data/tensor_opts.json")
        .expect("failed to read in tensor opts file");
    let opts: TensorOptions = serde_json::from_str(&opts_file_data)
        .expect("failed to parse tensor options");
    let (co, vals) = gentensor(opts, &world);
    assert_eq!(co[0].len(), vals.len());
    let nmodes = co.len();
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
    assert_eq!(co_set.len(), nnz);

    // Gather the nonzero counts onto to rank 0
    let root = world.process_at_rank(0);
    let nnz: Count = nnz.try_into().expect("failed to convert nnz count to MPI count");
    if rank == 0 {
        let mut counts: Vec<Count> = vec![0; size];
        root.gather_into_root(&nnz, &mut counts[..]);
        let disps: Vec<Count> = counts.iter().scan(0, |state, &count| {
            let disp = *state;
            *state += count;
            Some(disp)
        }).collect();
        let total_nnz = disps[disps.len() - 1] + counts[counts.len() - 1];
        let total_nnz: usize = total_nnz.try_into().expect("failed to convert nnz count");
        let mut complete_co = vec![];
        for m in 0..nmodes {
            let mut buffer: Vec<usize> = vec![0; total_nnz];
            let mut partition = PartitionMut::new(&mut buffer[..], &counts[..], &disps[..]);
            root.gather_varcount_into_root(&co[m], &mut partition);
            complete_co.push(buffer);
        }
        let mut complete_vals = vec![0.0; total_nnz];
        let mut partition = PartitionMut::new(&mut complete_vals[..], &counts[..], &disps[..]);
        root.gather_varcount_into_root(&vals, &mut partition);

        // Convert the tensor into a BTreeMap
        let mut tensor_data = BTreeMap::new();
        while complete_vals.len() > 0 {
            let mut new_co = vec![];
            for m in 0..nmodes {
                new_co.push(complete_co[m].pop().expect("missing coordinate"));
                complete_co[m].shrink_to_fit();
            }
            tensor_data.insert(new_co, complete_vals.pop().expect("missing value"));
            complete_vals.shrink_to_fit();
        }
        feat::analyze_tensor(&tensor_data);
    } else {
        root.gather_into(&nnz);
        for m in 0..nmodes {
            root.gather_varcount_into(&co[m]);
        }
        root.gather_varcount_into(&vals);
    }
}
