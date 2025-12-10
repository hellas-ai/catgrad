#include <stdio.h>    // for printf, fprintf: formatted input/output
#include <stdlib.h>   // for malloc, free, exit, and general utilities
#include <stdint.h>   // for fixed-width integer types like int64_t
#include <dlfcn.h>    // for dlopen, dlsym, dlclose: dynamic shared library loading


// Structure describing a rank-3 MLIR memref descriptor.
// Mirrors the layout produced by MLIR's LLVM lowering for tensor<3x1x4xf32>.
struct memref3d {
    float *allocated;   // Base pointer returned by malloc (or similar)
    float *aligned;     // Aligned pointer to the actual data start
    int64_t offset;     // Logical offset from aligned pointer to first element
    int64_t sizes[3];   // Extents of each dimension (here: [3,1,4])
    int64_t strides[3]; // Stride of each dimension in elements (row-major order)
};

// Structure for returning multiple memrefs
struct dual_memref_result {
    struct memref3d negated;    // The negated tensor
    struct memref3d unchanged;  // The unchanged input tensor copy
};

// Function pointer type matching the lowered MLIR symbol `@negate_f32`.
// It takes nine arguments encoding one input memref descriptor
// (two pointers + offset + sizes + strides) and returns a dual_memref_result struct.
typedef struct dual_memref_result (*negate_f32_t)(
    float*, float*, int64_t,int64_t,int64_t,
    int64_t,int64_t,int64_t,int64_t);

int main() {
    // Dynamically load the compiled shared object produced from MLIR.
    void *handle = dlopen("./main.so", RTLD_NOW);
    if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 1; }

    // Look up the MLIR-exported function by symbol name.
    negate_f32_t negate_f32 = (negate_f32_t)dlsym(handle, "negate_f32");
    if (!negate_f32) { fprintf(stderr, "%s\n", dlerror()); return 1; }

    // Define a static 3×1×4 tensor as the input buffer.
    float input[3][1][4] = {
        {{1,2,3,4}},
        {{5,6,7,8}},
        {{9,10,11,12}}
    };

    // Print the input tensor contents.
    printf("input...\n");
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 1; ++j) {
            for (int k = 0; k < 4; ++k) {
                printf("%6.2f ", input[i][j][k]);
            }
            printf("\n");
        }

    // Construct a memref descriptor describing the input buffer.
    struct memref3d in = {
        .allocated = (float*)input, // base and aligned point to same stack buffer
        .aligned   = (float*)input,
        .offset = 0,
        .sizes = {3,1,4},           // shape
        .strides = {4,4,1}          // standard row-major strides
    };

    // Invoke the MLIR function. It allocates and returns both the negated tensor
    // and a copy of the unchanged input tensor.
    struct dual_memref_result res = negate_f32(
        in.allocated, in.aligned, in.offset,
        in.sizes[0], in.sizes[1], in.sizes[2],
        in.strides[0], in.strides[1], in.strides[2]
    );

    // Print the negated output tensor from the returned memref.
    printf("negated output...\n");
    float *negated_data = res.negated.aligned;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 1; ++j) {
            for (int k = 0; k < 4; ++k)
                // Linearized index using known contiguous layout
                printf("%6.2f ", negated_data[i*4 + j*4 + k]);
            printf("\n");
        }

    // Print the unchanged output tensor from the returned memref.
    printf("unchanged output...\n");
    float *unchanged_data = res.unchanged.aligned;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 1; ++j) {
            for (int k = 0; k < 4; ++k)
                // Linearized index using known contiguous layout
                printf("%6.2f ", unchanged_data[i*4 + j*4 + k]);
            printf("\n");
        }

    // Free the heap memory allocated inside the MLIR function.
    printf("freeing res.negated...\n");
    free(res.negated.allocated);
    // Unload the shared object and exit.
    dlclose(handle);
    return 0;
}
