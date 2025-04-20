SPARSE_TENSOR_LIB = target/release/libsparse_tensor.so
SPARSE_TENSOR_LIB_NAME = sparse_tensor
SPARSE_TENSOR_FILES = src/lib.rs src/synthetic/mod.rs src/synthetic/c_api.rs
EXAMPLES = examples/ffi_test.c
EXAMPLE_BINS = $(EXAMPLES:.c=)
LDFLAGS = -L$(dir $(SPARSE_TENSOR_LIB)) -l$(SPARSE_TENSOR_LIB_NAME) -Wl,-rpath,$(shell realpath $(dir $(SPARSE_TENSOR_LIB)))
CFLAGS = -Iinclude

all: $(SPARSE_TENSOR_LIB) $(EXAMPLE_BINS)
.PHONY: all

$(SPARSE_TENSOR_LIB): $(SPARSE_TENSOR_FILES)
	cargo build --release

$(EXAMPLE_BINS): $(SPARSE_TENSOR_LIB)
$(EXAMPLE_BINS): %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

install: $(SPARSE_TENSOR_LIB)
	mkdir -p $(PREFIX)/lib && cp $(SPARSE_TENSOR_LIB) $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig && python3 scripts/generate_pc.py $(shell realpath $(PREFIX)) $(SPARSE_TENSOR_LIB_NAME) > $(PREFIX)/lib/pkgconfig/$(SPARSE_TENSOR_LIB_NAME).pc
	mkdir -p $(PREFIX)/include && cp include/*.h $(PREFIX)/include
.PHONY: all

clean:
	cargo clean && rm -rf $(EXAMPLE_BINS)
.PHONY: clean
