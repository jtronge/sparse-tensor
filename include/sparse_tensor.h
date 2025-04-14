#ifndef _SPARSE_TENSOR_H_
#define _SPARSE_TENSOR_H_

void *sparse_tensor_synthetic_options_load(char const *fname);
void sparse_tensor_synthetic_options_free(void *opts_handle);
void sparse_tensor_synthetic_generate(void *opts_handle);

#endif /* _SPARSE_TENSOR_H_ */
