#ifndef _SPARSE_TENSOR_H_
#define _SPARSE_TENSOR_H_

void *sparse_tensor_synthetic_options_load(char const *fname);
void sparse_tensor_synthetic_options_free(void *opts_handle);

/* Synthetic tensor in coo-form */
struct sparse_tensor_synthetic {
    /* Number of nonzeros */
    size_t nnz;
    /* 3 x nnz multi-dimensional array of entry coordinates */
    size_t *co[3];
    /* nnz double values for the tensor */
    double *vals;
};

/* Generate a synthetic tensor using the options handle */
struct sparse_tensor_synthetic *sparse_tensor_synthetic_generate(void *opts_handle, MPI_Comm comm);
/* Free the synthetic tensor */
void sparse_tensor_synthetic_free(struct sparse_tensor_synthetic *tensor);

#endif /* _SPARSE_TENSOR_H_ */
