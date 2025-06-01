//! Tensor feature analysis.
//!
//! Partially based on <https://arxiv.org/abs/2405.04944>
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use mpi::traits::*;
use sparse_tensor::{feat, TensorStream, SparseTensor};

const BUFFER_SIZE: usize = 8192;

fn main() {
    let universe = mpi::initialize().expect("failed to initialize MPI");
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    let mut local_co = vec![vec![]; 3];
    let mut local_values = vec![];
    if rank == 0 {
        let io_timer = Instant::now();
        let args: Vec<String> = std::env::args().map(|arg| arg.to_string()).collect();
        assert_eq!(args.len(), 2);
        let tensor_fname = &args[1];
        println!("loading tensor {}", tensor_fname);
        let file = File::open(tensor_fname).expect("failed to open tensor");
        let stream = BufReader::new(file);
        let mut loader = TensorStream::new(stream);
        let mut next_rank = 0;
        let mut done = false;
        while !done {
            // Copy BUFFER_SIZE values into a message
            let mut send_co = vec![vec![]; 3];
            let mut send_values = vec![];
            for _ in 0..BUFFER_SIZE {
                if let Some((co, value)) = loader.next() {
                    // Only save three modes, get rid of the rest
                    for m in 0..3 {
                        send_co[m].push(co[m]);
                    }
                    send_values.push(value);
                } else {
                    done = true;
                    break;
                }
            }

            // Send to next_rank or save locally
            if next_rank == 0 {
                local_values.extend(send_values.iter().map(|value| *value));
                for m in 0..3 {
                    local_co[m].extend(send_co[m].iter().map(|idx| *idx));
                }
            } else {
                // Send it to the next rank
                let dest = world.process_at_rank(next_rank);
                let count: isize = send_values.len().try_into().expect("failed to convert length to isize");
                dest.send(&count);
                dest.send(&send_values[..]);
                for m in 0..3 {
                    dest.send(&send_co[m][..]);
                }
            }

            next_rank = (next_rank + 1) % size;
        }

        // Send a stop message to all the other ranks
        for rank in 1..size {
            world.process_at_rank(rank).send(&0);
        }
        println!("io took {:.4} seconds", io_timer.elapsed().as_secs_f64());
    } else {
        let root = world.process_at_rank(0);
        loop {
            let mut count: isize = 0;
            root.receive_into(&mut count);
            if count == 0 {
                // Terminating condition
                break;
            }
            let count: usize = count.try_into().expect("failed to convert count into usize");
            let cur_len = local_values.len();
            local_values.resize(cur_len + count, 0.0);
            root.receive_into(&mut local_values[cur_len..]);
            for m in 0..3 {
                local_co[m].resize(cur_len + count, 0);
                root.receive_into(&mut local_co[m][cur_len..]);
            }
        }
    }
    let tensor = SparseTensor::new(local_values, local_co);
    feat::analyze_tensor(&tensor, &world);
}
