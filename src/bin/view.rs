//! Tool for viewing nonzeros of a 3D tensor.
use bzip2::read::BzDecoder;
use sparse_tensor::{get_tensor_dims_iter, TensorStream};
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter};
use std::path::Path;

fn write_svg<P: AsRef<Path>>(path: P, x_buckets: usize, y_buckets: usize, buckets: &[Vec<usize>]) {
    let f = File::create(path).expect("failed to open output file");
    let mut buf = BufWriter::new(f);
    let bucket_width = 4;
    let width = x_buckets * bucket_width;
    let height = y_buckets * bucket_width;
    writeln!(buf,
             "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{}\" height=\"{}\">",
             width, height)
        .expect("failed to write header");
    let max = *buckets
        .iter()
        .map(|counts| counts.iter().max().expect("missing inner max value"))
        .max()
        .expect("missing max value");
    for x in 0..x_buckets {
        for y in 0..y_buckets {
            let degree = (buckets[x][y] as f64) / (max as f64);
            let fill_value = ((1.0 - degree) * 255.0) as u8;
            let hex = format!("{:02x}", fill_value);
            let fill = format!("#{}{}{}", hex, hex, hex);
            // let fill = if x % 2 == 0 && y % 2 == 1 { "#00ff00" } else { "#000000" };
            writeln!(
                buf,
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\"/>",
                x * bucket_width,
                y * bucket_width,
                bucket_width,
                bucket_width,
                fill
            )
            .expect("failed to write svg rect tag");
        }
    }
    writeln!(buf, "</svg>").expect("failed to write last tag");
}

fn create_tensor_stream(fname: &str) -> impl Iterator<Item = (Vec<usize>, f64)> {
    let reader = if fname.ends_with(".bz2") {
        let f = File::open(fname).expect("failed to open tensor file");
        let bzf = BzDecoder::new(f);
        Box::new(BufReader::new(bzf)) as Box<dyn BufRead>
    } else {
        let f = File::open(fname).expect("failed to open tensor file");
        Box::new(BufReader::new(f)) as Box<dyn BufRead>
    };
    TensorStream::new(reader)
}

fn main() {
    let args: Vec<String> = std::env::args().map(|arg| arg.to_string()).collect();
    if args.len() != 3 {
        eprintln!("Usage: {} [TENSOR_FILE] [OUTPUT_FILE]", args[0]);
        return;
    }
    let tensor_fname = &args[1];
    let output_fname = &args[2];
    // Load the tensor and get the dimensions
    // let tensor_data = load_tensor(create_tensor_stream(tensor_fname));
    let tensor_dims = get_tensor_dims_iter(create_tensor_stream(tensor_fname));
    println!("{:?}", tensor_dims);
    let max_buckets = 40;
    // assert_eq!(tensor_dims.len(), 3);
    // TODO: This ignores the extreme tensor coordinate values
    let (x_buckets, y_buckets, nnz_per_bucket) = if tensor_dims[0] > tensor_dims[1] {
        let x_buckets = max_buckets;
        let nnz_per_bucket = tensor_dims[0] / x_buckets;
        let y_buckets = std::cmp::max(tensor_dims[1] / nnz_per_bucket, 1);
        (x_buckets, y_buckets, nnz_per_bucket)
    } else {
        let y_buckets = max_buckets;
        let nnz_per_bucket = tensor_dims[1] / y_buckets;
        let x_buckets = std::cmp::max(tensor_dims[0] / nnz_per_bucket, 1);
        (x_buckets, y_buckets, nnz_per_bucket)
    };
    println!("y_buckets: {}", y_buckets);
    println!("x_buckets: {}", x_buckets);
    println!("nnz_per_bucket: {}", nnz_per_bucket);
    assert!(y_buckets < 4000);
    let mut buckets = vec![vec![0; y_buckets + 1]; x_buckets + 1];
    for (co, _) in create_tensor_stream(tensor_fname) {
        let x = co[0] / nnz_per_bucket;
        let y = co[1] / nnz_per_bucket;
        if x > x_buckets || y > y_buckets {
            println!("co[0] = {}, co[1] = {}", co[0], co[1]);
            continue;
        }
        buckets[x][y] += 1;
    }
    println!("{:?}", buckets);
    write_svg(output_fname, x_buckets, y_buckets, &buckets);
}
