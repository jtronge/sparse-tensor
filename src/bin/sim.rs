use sparse_tensor::{count_nonzeros_per_slice, format_dims, get_tensor_dims, load_tensor};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;

fn get_primes(mut n: usize) -> Vec<usize> {
    let mut primes = vec![];
    while n != 1 {
        let mut next_prime = 1;
        for i in 2..=n {
            if (n % i) == 0 {
                next_prime = i;
                break;
            }
        }
        primes.push(next_prime);
        n /= next_prime;
    }
    primes
}

/// Compute the processor dimensions based on the layout.
fn compute_processor_dimensions(tensor_dims: &[usize], node_count: usize) -> Vec<usize> {
    // First compute a list of primes for the node_count number
    let primes = get_primes(node_count);
    let total_nnz: usize = tensor_dims.iter().sum();
    let target_nnz = total_nnz / node_count;
    let mut proc_dims: Vec<usize> = tensor_dims.iter().map(|_| 1).collect();
    for i in (0..primes.len()).rev() {
        // Find the mode (tensor dimension) contributing the most nonzeros
        let mut furthest = 0;
        let mut furthest_diff = 0;
        for mode in 0..tensor_dims.len() {
            let curr_nnz = tensor_dims[mode] / proc_dims[mode];
            let curr_diff = if curr_nnz > target_nnz {
                curr_nnz - target_nnz
            } else {
                0
            };
            if curr_diff > furthest_diff {
                furthest = mode;
                furthest_diff = curr_diff;
            }
        }
        proc_dims[furthest] *= primes[i];
    }
    proc_dims
}

fn find_layer_boundaries(
    tensor_data: &BTreeMap<Vec<usize>, f64>,
    tensor_dims: &[usize],
    proc_dims: &[usize],
    slice_sizes: &[Vec<usize>],
    mode: usize,
) -> Vec<usize> {
    let mut layer_ptrs: Vec<usize> = (0..proc_dims[mode] + 1).map(|_| 0).collect();
    layer_ptrs[0] = 0;
    layer_ptrs[proc_dims[mode]] = tensor_dims[mode];

    let layer_dim = proc_dims[mode];
    let mut layer_target_nnz = tensor_data.len() / layer_dim;

    if layer_dim != 1 {
        let mut curr_proc = 1;
        let mut last_nnz = 0;
        let mut nnzcount = slice_sizes[mode][0];
        for slice in 1..tensor_dims[mode] {
            if nnzcount >= (last_nnz + layer_target_nnz) {
                // Choose which slice to assign based on nonzero count diff
                let thisdist = nnzcount - (last_nnz + layer_target_nnz);
                let prevdist =
                    (last_nnz + layer_target_nnz) - (nnzcount - slice_sizes[mode][slice - 1]);
                last_nnz = if prevdist < thisdist {
                    nnzcount - slice_sizes[mode][slice - 1]
                } else {
                    nnzcount
                };

                layer_ptrs[curr_proc] = slice;
                curr_proc += 1;
                // Check if this is the last process
                if curr_proc == layer_dim {
                    break;
                }

                // Update target nonzeros per layer
                let denom = if layer_dim > (curr_proc - 1) {
                    layer_dim - (curr_proc - 1)
                } else {
                    1
                };
                layer_target_nnz = (tensor_data.len() - last_nnz) / denom;
            }
            nnzcount += slice_sizes[mode][slice];
        }
    }
    layer_ptrs
}

/// Generate the processor coordinates for the number nodes and processor grid.
fn generate_proc_coords(node_count: usize, proc_dims: &[usize]) -> Vec<Vec<usize>> {
    let mut coords = vec![];
    let mut tmp_co: Vec<usize> = (0..proc_dims.len()).map(|_| 0).collect();
    for _ in 0..node_count {
        coords.push(tmp_co.to_vec());
        assert_eq!(proc_dims.len(), tmp_co.len());
        // Increment a processor coordinate to the next possible value
        for mode1 in (0..proc_dims.len()).rev() {
            if (tmp_co[mode1] + 1) < proc_dims[mode1] {
                tmp_co[mode1] += 1;
                // Reset later coordinates
                for mode2 in mode1 + 1..proc_dims.len() {
                    tmp_co[mode2] = 0;
                }
                break;
            }
        }
    }
    coords
}

#[derive(Debug)]
struct Process {
    /// Process rank
    rank: usize,

    /// Process coordinate
    co: Vec<usize>,

    /// Layer starts for each mode
    layer_starts: Vec<usize>,

    /// Layer ends for each mode
    layer_ends: Vec<usize>,
}

fn create_procs(
    node_count: usize,
    tensor_dims: &[usize],
    layer_ptrs: &[Vec<usize>],
    proc_coords: &[Vec<usize>],
) -> Vec<Process> {
    let mut procs = vec![];
    for rank in 0..node_count {
        let co = proc_coords[rank].to_vec();
        let layer_starts = (0..tensor_dims.len())
            .map(|mode| layer_ptrs[mode][co[mode]])
            .collect();
        let layer_ends = (0..tensor_dims.len())
            .map(|mode| layer_ptrs[mode][co[mode] + 1])
            .collect();
        procs.push(Process {
            rank,
            co,
            layer_starts,
            layer_ends,
        });
    }
    procs
}

fn distribute_factor_matrix_rows(_procs: &[Process]) {
        /*
    for proc in procs {
                // Get the max dimension for this layer
                let max_dim = (0..proc.layer_starts.len())
                    .map(|mode| proc.layer_ends[mode] - proc.layer_starts[mode])
                    .max();

                // Count appearances of indices across all ranks
                let pcount: Vec<usize> = (0..max_dim).map(|_| 0).collect();
                let local: Vec<usize> = (0..max_dim).map(|_| 0).collect();
    }
        */
}

fn main() {
    // Parse args
    let args: Vec<String> = std::env::args().map(|arg| arg.to_string()).collect();
    if args.len() != 3 {
        eprintln!("Usage: {} [TENSOR_FILE] [NODE_COUNT]", args[0]);
        return;
    }
    let tensor_fname = &args[1];
    let node_count = usize::from_str_radix(&args[2], 10)
        .expect("failed to parse command line argument for node count");
    // Load the tensor data
    let tensor_file = File::open(tensor_fname).expect("failed to open tensor file");
    let tensor_reader = BufReader::new(tensor_file);
    let tensor_data = load_tensor(tensor_reader);
    let tensor_dims = get_tensor_dims(&tensor_data);
    println!("=> tensor dimensions = {}", format_dims(&tensor_dims));

    // Compute the processor grid and layout
    let proc_dims = compute_processor_dimensions(&tensor_dims, node_count);
    println!("=> process grid = {}", format_dims(&proc_dims));

    // Now count the number of nonzeros per slice
    let slice_sizes = count_nonzeros_per_slice(&tensor_data, &tensor_dims);
    assert_eq!(slice_sizes.len(), tensor_dims.len());

    let mut layer_ptrs = vec![];
    for mode in 0..tensor_dims.len() {
        layer_ptrs.push(find_layer_boundaries(
            &tensor_data,
            &tensor_dims,
            &proc_dims,
            &slice_sizes,
            mode,
        ));
    }

    let proc_coords = generate_proc_coords(node_count, &proc_dims);
    assert_eq!(proc_coords.len(), node_count);

    let procs = create_procs(node_count, &tensor_dims, &layer_ptrs, &proc_coords);

    // Now distribute the factor matrices
    distribute_factor_matrix_rows(&procs);
}
