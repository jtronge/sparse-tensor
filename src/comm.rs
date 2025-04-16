//! Parallel communication module and abstraction.
use std::os::raw::c_int;
use mpi_sys::{MPI_Init, MPI_Finalize, MPI_Comm, MPI_Comm_size, MPI_Comm_rank, MPI_Allreduce, MPI_Op, RSMPI_SUM, RSMPI_UINT64_T, RSMPI_COMM_WORLD};

pub struct Environment {
    world: Comm,
}

impl Environment {
    /// Initialize the environment.
    pub unsafe fn init() -> Environment {
        MPI_Init(std::ptr::null_mut(), std::ptr::null_mut());
        Environment {
            world: unsafe { Comm::from_raw(RSMPI_COMM_WORLD) },
        }
    }

    /// Return the world communicator.
    pub fn comm_world(&self) -> &Comm {
        &self.world
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        unsafe {
            MPI_Finalize();
        }
    }
}

/// Wrapper around an MPI_Comm.
pub struct Comm(MPI_Comm);

impl Comm {
    /// Create a new communicator from the raw MPI_Comm.
    ///
    /// SAFETY: The calling code must ensure that this communicator is valid.
    pub unsafe fn from_raw(comm: MPI_Comm) -> Comm {
        Comm(comm)
    }

    /// Return the communicator size --- the number of processes in the communicator.
    pub fn size(&self) -> usize {
        let mut size: c_int = 0;
        unsafe { MPI_Comm_size(self.0, &mut size) };
        size.try_into().expect("failed to convert size to usize")
    }

    /// Return the rank of this process.
    pub fn rank(&self) -> usize {
        let mut rank: c_int = 0;
        unsafe { MPI_Comm_rank(self.0, &mut rank) };
        rank.try_into().expect("failed to convert rank to usize")
    }

    /// Perform an allreduce.
    ///
    /// TODO: Make this more generic
    pub fn allreduce(&self, sendbuf: &u64, recvbuf: &mut u64, op: Operation) {
        unsafe {
            MPI_Allreduce((sendbuf as *const _) as *const _, (recvbuf as *mut _) as *mut _, 1,
                          // RSMPI_UINT64_T, RSMPI_SUM, comm);
                          RSMPI_UINT64_T, op.raw_op(), self.0);
        }
    }
}

/// A binary associative operation that can be used with reduce, scan, etc.
pub enum Operation {
    /// Compute a sum of values.
    Sum,
}

impl Operation {
    unsafe fn raw_op(&self) -> MPI_Op {
        match self {
            &Operation::Sum => RSMPI_SUM,
        }
    }
}
