use sparse_tensor::SparseTensor;
use sparse_tensor::synthetic::{TensorOptions, gentensor};
use sparse_tensor::feat::analyze_tensor;

fn main() {
    let universe = mpi::initialize().expect("failed to initialize MPI");
    let world = universe.world();

    let args: Vec<String> = std::env::args().map(|s| s.to_string()).collect();
    let opts_fname = &args[1];
    let file_data = std::fs::read_to_string(opts_fname).expect("missing file data");
    let tensor_opts: TensorOptions = serde_json::from_str(&file_data).expect("failed to parse tensor options");
    let tensor = gentensor(tensor_opts, &world);
    analyze_tensor(&tensor, &world);
}
