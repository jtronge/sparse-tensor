#include <stdlib.h>
#include <assert.h>
#include "sparse_tensor.h"

int main(void)
{
    void *opts = sparse_tensor_synthetic_options_load("data/tensor_opts.json");
    assert(opts != NULL);
    sparse_tensor_synthetic_options_free(opts);
    return 0;
}
