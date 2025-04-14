SPARSE_TENSOR_LIB = target/release/libsparse_tensor.so
SPARSE_TENSOR_LIB_NAME = sparse_tensor
SPARSE_TENSOR_FILES = src/lib.rs src/synthetic.rs
EXAMPLES = examples/ffi_test.c
EXAMPLE_BINS = $(EXAMPLES:.c=)
LDFLAGS = -L$(dir $(SPARSE_TENSOR_LIB)) -l$(SPARSE_TENSOR_LIB_NAME) -Wl,-rpath,$(shell realpath $(dir $(SPARSE_TENSOR_LIB)))
CFLAGS = -Iinclude

all: $(SPARSE_TENSOR_LIB) $(EXAMPLE_BINS)
.PHONY: all

$(EXAMPLE_BINS): $(SPARSE_TENSOR_LIB)
$(EXAMPLE_BINS): %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	cargo clean && rm $(EXAMPLE_BINS)
.PHONY: clean
