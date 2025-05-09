//! C API submodule for the sparse tensor generator.
use std::os::raw::{c_char, c_void};
use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::time::Instant;
use std::alloc::{alloc, dealloc, Layout};
use mpi::topology::SimpleCommunicator;
use mpi::raw::FromRaw;
use mpi::ffi::{MPI_Comm, MPI_Comm_dup};
use mpi::traits::*;

use crate::synthetic::{TensorOptions, gentensor};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sparse_tensor_synthetic_options_load(fname: *const c_char) -> *mut c_void {
    let fname = CStr::from_ptr(fname).to_string_lossy().to_string();
    let file_text_result = std::fs::read_to_string(&fname);
    if file_text_result.is_err() {
        eprintln!("sparse_tensor: failed to open {}", fname);
        return std::ptr::null_mut();
    }
    let file_text = file_text_result.expect("should not be possible");
    let opts: TensorOptions = serde_json::from_str(&file_text).expect("failed to parse tensor options");
    Box::into_raw(Box::new(opts)) as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sparse_tensor_synthetic_options_free(opts_handle: *mut c_void) {
    let _ = Box::from_raw(opts_handle as *mut TensorOptions);
}

#[repr(C)]
pub struct SyntheticTensor {
    /// Number of nonzeros.
    nnz: usize,
    /// Entry coordinates.
    co: [*mut usize; 3],
    /// Values for each nonzero.
    vals: *mut f64,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sparse_tensor_synthetic_generate(
    opts_handle: *mut c_void,
    comm: MPI_Comm,
) -> *mut SyntheticTensor {
    let tensor_opts = opts_handle as *mut TensorOptions;

    // We have to duplicate the communicator due to RSMPI trying to call
    // MPI_Comm_free() on it (see https://github.com/rsmpi/rsmpi/issues/32)
    let mut tmp_comm = MaybeUninit::uninit();
    MPI_Comm_dup(comm, tmp_comm.as_mut_ptr());
    let tmp_comm = tmp_comm.assume_init();

    let rsmpi_comm = SimpleCommunicator::from_raw(tmp_comm);
    let tensor = gentensor((*tensor_opts).clone(), &rsmpi_comm);
    let co = tensor.co;
    let vals = tensor.values;

    let now = Instant::now();
    // Some ugly manual mem to work properly with C
    assert_eq!(co[0].len(), vals.len());
    let local_nnz = co[0].len();
    let val_layout = Layout::array::<f64>(local_nnz)
        .expect("failed to create memory layout for tensor values");
    let val_ptr = alloc(val_layout);
    std::ptr::copy_nonoverlapping(vals.as_ptr(), val_ptr as *mut _, local_nnz);
    assert_ne!(val_ptr, std::ptr::null_mut());
    let co_layout = Layout::array::<usize>(local_nnz)
        .expect("failed to create memory layout for tensor coordinates");
    let co_ptrs: Vec<*mut usize> = (0..3).map(|i| {
        let co_ptr = alloc(co_layout);
        assert_ne!(co_ptr, std::ptr::null_mut());
        std::ptr::copy_nonoverlapping(co[i].as_ptr(), co_ptr as *mut _, local_nnz);
        co_ptr as *mut _
    }).collect();
    let stensor = SyntheticTensor {
        nnz: local_nnz,
        co: co_ptrs[..].try_into().expect("failed to convert from vec to array in struct"),
        vals: val_ptr as *mut _,
    };
    if rsmpi_comm.rank() == 0 {
        println!("==> mem_alloc_time={}", now.elapsed().as_secs_f64());
    }
    Box::into_raw(Box::new(stensor))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sparse_tensor_synthetic_free(stensor: *mut SyntheticTensor) {
    let stensor = Box::from_raw(stensor);
    let nnz = stensor.nnz;
    let val_layout = Layout::array::<f64>(nnz)
        .expect("failed to create memory layout for tensor values");
    dealloc(stensor.vals as *mut _, val_layout);
    let co_layout = Layout::array::<usize>(nnz)
        .expect("failed to create memory layout for tensor coordinates");
    for co_ptr in stensor.co {
        dealloc(co_ptr as *mut _, co_layout);
    }
}
