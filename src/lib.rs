use std::collections::BTreeMap;
use std::io::prelude::*;
use std::io::{BufReader, Read};
use std::str::FromStr;

pub mod synthetic;
pub mod feat;

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

/// Sparse tensor data structure.
pub struct SparseTensor {
    /// Tensor values.
    pub values: Vec<f64>,

    /// nmodes x n array of coordinate indices in the tensor.
    pub co: Vec<Vec<usize>>,
}

impl SparseTensor {
    pub fn new(values: Vec<f64>, co: Vec<Vec<usize>>) -> SparseTensor {
        for co_list in &co {
            assert_eq!(values.len(), co_list.len());
        }
        SparseTensor {
            values,
            co,
        }
    }

    /// Return the number of nonzeros.
    #[inline]
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Return the number of modes.
    #[inline]
    pub fn modes(&self) -> usize {
        self.co.len()
    }

    /// Return the dimensions of the tensor.
    pub fn tensor_dims(&self) -> Vec<usize> {
        let mut tensor_dims = vec![];
        for i in 0..self.values.len() {
            if i == 0 {
                for m in 0..self.modes() {
                    tensor_dims.push(self.co[m][i]);
                }
            }
            for m in 0..self.modes() {
                tensor_dims[m] = std::cmp::max(tensor_dims[m], self.co[m][i]);
            }
        }
        for m in 0..tensor_dims.len() {
            tensor_dims[m] += 1;
        }
        assert_eq!(tensor_dims.len(), self.modes());
        tensor_dims
    }

    /// Sort the nonzeros by the values of a specific mode.
    pub fn sort_by_mode(&mut self, mode: usize) {
        let mut perm: Vec<usize> = (0..self.count()).collect();
        perm.sort_by_key(|&i| self.co[mode][i]);
        for i in 0..self.count() {
            // Follow the previous permutations made
            let mut dest = perm[i];
            while dest < i {
                dest = perm[dest];
            }
            for m in 0..self.modes() {
                let tmp = self.co[m][i];
                self.co[m][i] = self.co[m][dest];
                self.co[m][dest] = tmp;
            }
            let tmp = self.values[i];
            self.values[i] = self.values[dest];
            self.values[dest] = tmp;
        }
    }
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
    tensor: &SparseTensor,
    tensor_dims: &[usize],
) -> Vec<Vec<usize>> {
    let mut slice_sizes = vec![];
    for mode in 0..tensor_dims.len() {
        let sizes: Vec<usize> = (0..tensor_dims[mode]).map(|_| 0).collect();
        slice_sizes.push(sizes);
    }
    for i in 0..tensor.count() {
        for mode in 0..tensor.modes() {
            slice_sizes[mode][tensor.co[mode][i]] += 1;
        }
    }
    slice_sizes
}
