use std::collections::BTreeMap;
use std::io::prelude::*;
use std::io::{BufReader, Read};
use std::str::FromStr;

pub struct TensorStream<R: BufRead> {
    stream: R,
    line_buf: String,
}

impl<R: BufRead> TensorStream<R> {
    pub fn new(stream: R) -> TensorStream<R> {
        TensorStream {
            stream,
            line_buf: String::new(),
        }
    }
}

impl<R: BufRead> Iterator for TensorStream<R> {
    type Item = (Vec<usize>, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while let Ok(size) = self.stream.read_line(&mut self.line_buf) {
            if size == 0 {
                // EOF
                break;
            }
            let line = self.line_buf.trim();
            // Skip comments
            if line.starts_with("#") {
                continue;
            }
            let parts: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            assert!(parts.len() >= 2);
            println!("parts: {:?}", parts);
            // Load coordinates
            let co: Vec<usize> = parts[..parts.len() - 1]
                .iter()
                // Make sure to subtract 1 so we have 0-based indexing
                .map(|s| usize::from_str_radix(s, 10).expect("failed to parse tensor coordinate") - 1)
                .collect();
            let value = f64::from_str(&parts[parts.len() - 1]).expect("failed to parse tensor value");
            self.line_buf.clear();
            return Some((co, value));
        }

        None
    }
}

/// Load a tensor from a file.
pub fn load_tensor<R: Read>(stream: R) -> BTreeMap<Vec<usize>, f64> {
    let stream = BufReader::new(stream);
    let loader = TensorStream::new(stream);
    let mut tensor_data = BTreeMap::new();
    for (co, value) in loader {
        tensor_data.insert(co, value);
    }
    tensor_data
}

/// Get the dimensions of a tensor using an iterator over the the nonzeros
pub fn get_tensor_dims_iter(tensor_iter: impl Iterator<Item = (Vec<usize>, f64)>) -> Vec<usize> {
    let mut tensor_dims = vec![];
    for (i, (co, _)) in tensor_iter.enumerate() {
        if i == 0 {
            tensor_dims.extend(&co[..]);
        }
        assert_eq!(co.len(), tensor_dims.len());
        for j in 0..co.len() {
            tensor_dims[j] = std::cmp::max(tensor_dims[j], co[j]);
        }
    }
    for j in 0..tensor_dims.len() {
        tensor_dims[j] += 1;
    }
    assert!(tensor_dims.len() > 0);
    tensor_dims
}

/// Get the dimensions of the tensor.
pub fn get_tensor_dims(tensor_data: &BTreeMap<Vec<usize>, f64>) -> Vec<usize> {
    // TODO: This is not the efficient -- an allocation occurs for each coordinate here
    get_tensor_dims_iter(tensor_data.iter().map(|(co, value)| (co.to_vec(), *value)))
}

/// Return a string for representing the dimensions.
pub fn format_dims(tensor_dims: &[usize]) -> String {
    let mut s = String::new();
    for (i, dim) in tensor_dims.iter().enumerate() {
        if i > 0 {
            s.push('x');
        }
        s.push_str(&format!("{}", dim));
    }
    s
}

/// Count the number of nonzeros in each slice.
///
/// Returns a vector of vector of counts, slice_sizes, such that
/// slice_sizes[mode][slice] is the number of nonzeros for the indexed mode
/// and slice.
pub fn count_nonzeros_per_slice(
    tensor_data: &BTreeMap<Vec<usize>, f64>,
    tensor_dims: &[usize],
) -> Vec<Vec<usize>> {
    let mut slice_sizes = vec![];
    for mode in 0..tensor_dims.len() {
        let sizes: Vec<usize> = (0..tensor_dims[mode]).map(|_| 0).collect();
        slice_sizes.push(sizes);
    }
    for (co, _) in tensor_data {
        for mode in 0..co.len() {
            slice_sizes[mode][co[mode]] += 1;
        }
    }
    slice_sizes
}
