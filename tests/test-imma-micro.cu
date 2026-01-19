// Micro test harness for IMMA INT4 tensor core debugging
// Tests mma.sync.m16n8k32.s4.s4 step by step
//
// Build: nvcc -arch=sm_86 -o test-imma-micro test-imma-micro.cu
// Run: ./test-imma-micro

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// MMA PTX wrapper for m16n8k32 s4.s4
// ============================================================================
// A: 16x32 row-major signed 4-bit
// B: 32x8 col-major signed 4-bit
// C/D: 16x8 row-major signed 32-bit
//
// Per-thread fragment sizes:
// A: 2 x uint32 (16 s4 values)
// B: 1 x uint32 (8 s4 values)
// C/D: 4 x int32 (4 values)

__device__ __forceinline__ void mma_m16n8k32_s4(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1,
    uint32_t b0,
    int32_t c0, int32_t c1, int32_t c2, int32_t c3
) {
#if __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1),
          "r"(b0),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
#else
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
    printf("MMA not supported on this architecture!\n");
#endif
}

// ============================================================================
// Test 1: Verify basic MMA with known values (all ones)
// ============================================================================
// If A[i][k] = 1 for all i,k and B[k][j] = 1 for all k,j
// Then C[i][j] = sum_k(1*1) = 32 for all i,j

__global__ void test_mma_ones_kernel(int32_t* C_out) {
    const int lane = threadIdx.x % 32;
    
    // Fill A fragment with all 1s (signed 4-bit: 0x1 repeated)
    // Each uint32 holds 8 s4 values, so 0x11111111 = [1,1,1,1,1,1,1,1]
    uint32_t a0 = 0x11111111;  // 8 ones
    uint32_t a1 = 0x11111111;  // 8 ones (for second half of rows)
    
    // Fill B fragment with all 1s
    uint32_t b0 = 0x11111111;  // 8 ones
    
    // Zero accumulator
    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int32_t d0, d1, d2, d3;
    
    mma_m16n8k32_s4(d0, d1, d2, d3, a0, a1, b0, c0, c1, c2, c3);
    
    // Store results (only lane 0 for simplicity)
    if (lane == 0) {
        C_out[0] = d0;
        C_out[1] = d1;
        C_out[2] = d2;
        C_out[3] = d3;
    }
    
    // All lanes store their results for analysis
    C_out[4 + lane * 4 + 0] = d0;
    C_out[4 + lane * 4 + 1] = d1;
    C_out[4 + lane * 4 + 2] = d2;
    C_out[4 + lane * 4 + 3] = d3;
}

void test_mma_ones() {
    printf("\n=== Test 1: MMA with all ones ===\n");
    printf("Expected: Each output element = 32 (sum of 32 ones)\n\n");
    
    int32_t* d_C;
    int32_t h_C[4 + 32*4];
    
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(h_C)));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeof(h_C)));
    
    test_mma_ones_kernel<<<1, 32>>>(d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost));
    
    printf("Lane 0 results: d0=%d, d1=%d, d2=%d, d3=%d\n", h_C[0], h_C[1], h_C[2], h_C[3]);
    
    printf("\nAll lanes' results:\n");
    for (int lane = 0; lane < 32; lane++) {
        int32_t* vals = &h_C[4 + lane * 4];
        printf("  Lane %2d: d0=%4d, d1=%4d, d2=%4d, d3=%4d\n", 
               lane, vals[0], vals[1], vals[2], vals[3]);
    }
    
    cudaFree(d_C);
}

// ============================================================================
// Test 2: Verify MMA output layout mapping
// ============================================================================
// Set A[i][k] = i (row index) for all k
// Set B[k][j] = 1 for all k,j
// Then C[i][j] = sum_k(i * 1) = i * 32

__global__ void test_mma_row_index_kernel(int32_t* C_out, int8_t* A_matrix, int8_t* B_matrix) {
    const int lane = threadIdx.x % 32;
    
    // We need to figure out which A elements each lane loads
    // For now, let's set up A matrix in global memory and load properly
    
    // A is 16x32 row-major s4
    // B is 32x8 col-major s4
    
    // Load A fragment - this is the tricky part
    // According to PTX docs, for m16n8k32 row.col:
    // Thread (lane) mapping for A fragment (2 registers, 16 s4 each):
    // Register a0: elements for rows [0,8) at specific K positions
    // Register a1: elements for rows [8,16) at specific K positions
    
    // Let's load based on expected layout
    // For simplicity, pack A where each row = row_index repeated 32 times
    
    uint32_t a0 = 0, a1 = 0;
    
    // Each thread loads 8 s4 values into a0 and 8 into a1
    // The mapping is: lane determines which K positions and rows
    // Simplified: just read from prepared matrix
    
    // Thread lane -> (row_group, k_offset)
    // For row-major A: lane/4 gives row offset mod 8, lane%4 gives k_offset / 8
    int row_offset = lane / 4;  // 0-7
    int k_base = (lane % 4) * 8;  // 0, 8, 16, 24
    
    // Pack 8 s4 values from row (row_offset) starting at column k_base
    uint32_t packed = 0;
    for (int i = 0; i < 8; i++) {
        int8_t val = A_matrix[row_offset * 32 + k_base + i];
        packed |= ((uint32_t)(val & 0xF)) << (i * 4);
    }
    a0 = packed;
    
    // Same for rows 8-15
    packed = 0;
    for (int i = 0; i < 8; i++) {
        int8_t val = A_matrix[(8 + row_offset) * 32 + k_base + i];
        packed |= ((uint32_t)(val & 0xF)) << (i * 4);
    }
    a1 = packed;
    
    // Load B fragment (32x8 col-major)
    // Each thread loads 8 s4 values
    // Thread mapping: lane%8 -> column, lane/8 -> row_group
    int col = lane % 8;
    int row_base = (lane / 8) * 8;  // 0, 8, 16, 24
    
    packed = 0;
    for (int i = 0; i < 8; i++) {
        // Col-major: B[row][col] is at B_matrix[col * 32 + row]
        int8_t val = B_matrix[col * 32 + row_base + i];
        packed |= ((uint32_t)(val & 0xF)) << (i * 4);
    }
    uint32_t b0 = packed;
    
    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int32_t d0, d1, d2, d3;
    
    mma_m16n8k32_s4(d0, d1, d2, d3, a0, a1, b0, c0, c1, c2, c3);
    
    // Store all results
    C_out[lane * 4 + 0] = d0;
    C_out[lane * 4 + 1] = d1;
    C_out[lane * 4 + 2] = d2;
    C_out[lane * 4 + 3] = d3;
}

void test_mma_row_index() {
    printf("\n=== Test 2: MMA with A[i][k]=i, B[k][j]=1 ===\n");
    printf("Expected: C[i][j] = i * 32\n\n");
    
    int8_t h_A[16 * 32];  // 16x32 row-major
    int8_t h_B[32 * 8];   // 32x8 col-major
    int32_t h_C[32 * 4];
    
    // Fill A: A[i][k] = i (clamped to s4 range [-8,7])
    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 32; k++) {
            h_A[i * 32 + k] = (int8_t)(i < 8 ? i : 7);  // clamp to s4 max
        }
    }
    
    // Fill B: B[k][j] = 1
    for (int k = 0; k < 32; k++) {
        for (int j = 0; j < 8; j++) {
            h_B[j * 32 + k] = 1;  // col-major
        }
    }
    
    int8_t* d_A;
    int8_t* d_B;
    int32_t* d_C;
    
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(h_A)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(h_B)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(h_C)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeof(h_C)));
    
    test_mma_row_index_kernel<<<1, 32>>>(d_C, d_A, d_B);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost));
    
    printf("Per-lane MMA outputs (d0, d1, d2, d3):\n");
    for (int lane = 0; lane < 32; lane++) {
        int32_t* vals = &h_C[lane * 4];
        printf("  Lane %2d: %4d, %4d, %4d, %4d\n", 
               lane, vals[0], vals[1], vals[2], vals[3]);
    }
    
    // Compute expected result with CPU
    printf("\nExpected C matrix (i * 32 for each row i):\n");
    int32_t expected[16][8];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            int32_t sum = 0;
            for (int k = 0; k < 32; k++) {
                sum += (int32_t)h_A[i * 32 + k] * (int32_t)h_B[j * 32 + k];
            }
            expected[i][j] = sum;
        }
    }
    for (int i = 0; i < 16; i++) {
        printf("  Row %2d:", i);
        for (int j = 0; j < 8; j++) {
            printf(" %4d", expected[i][j]);
        }
        printf("\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Test 3: Discover exact thread-to-output mapping
// ============================================================================
// Each thread writes unique values to identify which output positions it owns

__global__ void test_mma_layout_discovery_kernel(int32_t* C_out) {
    const int lane = threadIdx.x % 32;
    
    // Use identity-ish values: each output should be (lane * 1000 + register_index)
    // We use accumulator input to tag outputs
    
    uint32_t a0 = 0, a1 = 0;  // zeros
    uint32_t b0 = 0;          // zeros
    
    // Set accumulator to lane-specific values
    int32_t c0 = lane * 1000 + 0;
    int32_t c1 = lane * 1000 + 1;
    int32_t c2 = lane * 1000 + 2;
    int32_t c3 = lane * 1000 + 3;
    
    int32_t d0, d1, d2, d3;
    
    mma_m16n8k32_s4(d0, d1, d2, d3, a0, a1, b0, c0, c1, c2, c3);
    
    // D = A*B + C, but A*B = 0, so D = C
    // This tells us exactly which lane/register maps to which output
    
    C_out[lane * 4 + 0] = d0;
    C_out[lane * 4 + 1] = d1;
    C_out[lane * 4 + 2] = d2;
    C_out[lane * 4 + 3] = d3;
}

void test_mma_layout_discovery() {
    printf("\n=== Test 3: Discover thread-to-output mapping ===\n");
    printf("Each thread's C registers tagged with lane*1000+reg_idx\n");
    printf("Output shows which lane/register owns each C[i][j]\n\n");
    
    int32_t* d_C;
    int32_t h_C[32 * 4];
    
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(h_C)));
    
    test_mma_layout_discovery_kernel<<<1, 32>>>(d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost));
    
    printf("Raw per-lane outputs:\n");
    for (int lane = 0; lane < 32; lane++) {
        int32_t* vals = &h_C[lane * 4];
        printf("  Lane %2d: d0=%5d, d1=%5d, d2=%5d, d3=%5d\n", 
               lane, vals[0], vals[1], vals[2], vals[3]);
    }
    
    // Build reverse mapping: for each (row, col), which lane/reg?
    printf("\nReconstructed C[16][8] matrix ownership:\n");
    printf("Format: lane.reg\n\n");
    
    // The MMA output C is 16x8
    // We need to figure out the mapping from lane/reg -> (row, col)
    // From PTX docs for m16n8k32:
    // groupID = token / 4 (where token = lane)
    // threadID_in_group = token % 4
    //
    // For C fragment:
    // c0 -> (groupID, threadID_in_group * 2)
    // c1 -> (groupID, threadID_in_group * 2 + 1)
    // c2 -> (groupID + 8, threadID_in_group * 2)
    // c3 -> (groupID + 8, threadID_in_group * 2 + 1)
    
    printf("According to PTX spec mapping:\n");
    for (int lane = 0; lane < 32; lane++) {
        int groupID = lane / 4;
        int tid_in_group = lane % 4;
        
        int row0 = groupID;
        int row1 = groupID + 8;
        int col0 = tid_in_group * 2;
        int col1 = tid_in_group * 2 + 1;
        
        printf("  Lane %2d: d0->C[%2d][%d], d1->C[%2d][%d], d2->C[%2d][%d], d3->C[%2d][%d]\n",
               lane, row0, col0, row0, col1, row1, col0, row1, col1);
    }
    
    cudaFree(d_C);
}

// ============================================================================
// Test 3.4: Empirically discover B fragment layout using identity matrix
// ============================================================================
// Set A = all 1s, B = identity-like pattern where B[k][j] = (k == j*4) ? 1 : 0
// This way we can see exactly which K positions contribute to which output columns

__global__ void test_discover_b_layout_kernel(
    const int8_t* B,  // 32x8 col-major
    int32_t* per_thread_results,
    int layout_variant  // Try different layouts
) {
    const int lane = threadIdx.x % 32;
    
    // A = all 1s
    uint32_t a0 = 0x11111111;
    uint32_t a1 = 0x11111111;
    
    uint32_t b0 = 0;
    
    if (layout_variant == 0) {
        // Layout 0: lane % 8 -> col, lane / 8 -> k_group (original)
        int b_col = lane % 8;
        int b_k_base = (lane / 8) * 8;
        for (int i = 0; i < 8; i++) {
            int8_t val = B[b_col * 32 + b_k_base + i];
            b0 |= ((uint32_t)(val & 0xF)) << (i * 4);
        }
    } else if (layout_variant == 1) {
        // Layout 1: lane / 4 -> k_group, lane % 4 -> col_pair (interleaved)
        int groupID = lane / 4;
        int tid = lane % 4;
        int b_k_base = groupID * 4;
        int b_col = tid * 2;
        for (int i = 0; i < 4; i++) {
            int8_t val0 = B[b_col * 32 + b_k_base + i];
            int8_t val1 = B[(b_col + 1) * 32 + b_k_base + i];
            b0 |= ((uint32_t)(val0 & 0xF)) << (i * 4);
            b0 |= ((uint32_t)(val1 & 0xF)) << ((i + 4) * 4);
        }
    } else if (layout_variant == 2) {
        // Layout 2: Based on PTX doc - B fragment for col-major 32x8
        // 32 threads load 32 uint32 (each = 8 s4) in specific pattern
        // Try: each thread loads 8 consecutive K values for ONE column
        // lane determines both column and which 8 K values
        int b_col = lane / 4;  // 0-7 (8 columns)
        int b_k_base = (lane % 4) * 8;  // 0, 8, 16, 24
        for (int i = 0; i < 8; i++) {
            int8_t val = B[b_col * 32 + b_k_base + i];
            b0 |= ((uint32_t)(val & 0xF)) << (i * 4);
        }
    } else if (layout_variant == 3) {
        // Layout 3: Reverse of layout 2
        int b_k_base = (lane / 8) * 8;  // 0, 8, 16, 24
        int b_col = lane % 8;  // 0-7
        for (int i = 0; i < 8; i++) {
            int8_t val = B[b_col * 32 + b_k_base + i];
            b0 |= ((uint32_t)(val & 0xF)) << (i * 4);
        }
    }
    
    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int32_t d0, d1, d2, d3;
    
    mma_m16n8k32_s4(d0, d1, d2, d3, a0, a1, b0, c0, c1, c2, c3);
    
    per_thread_results[lane * 4 + 0] = d0;
    per_thread_results[lane * 4 + 1] = d1;
    per_thread_results[lane * 4 + 2] = d2;
    per_thread_results[lane * 4 + 3] = d3;
}

void test_discover_b_layout() {
    printf("\n=== Test 3.4: Discover B fragment layout ===\n");
    
    int8_t h_B[32 * 8];  // 32x8 col-major
    
    // Fill B: B[k][j] = min(k, 7) for all j
    for (int k = 0; k < 32; k++) {
        for (int j = 0; j < 8; j++) {
            h_B[j * 32 + k] = (int8_t)(k < 8 ? k : 7);
        }
    }
    
    // Expected: C[i][j] = sum over k of (1 * B[k][j]) = sum of min(k,7) for k=0..31
    // = 0+1+2+3+4+5+6+7 + 7*24 = 28 + 168 = 196
    printf("With A=all 1s, B[k][j] = min(k,7):\n");
    printf("Expected: C[i][j] = 196 for all i,j\n\n");
    
    int8_t* d_B;
    int32_t* d_results;
    int32_t h_results[32 * 4];
    
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(h_B)));
    CHECK_CUDA(cudaMalloc(&d_results, sizeof(h_results)));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    
    // Try all layout variants
    for (int variant = 0; variant <= 3; variant++) {
        printf("--- Layout variant %d ---\n", variant);
        
        test_discover_b_layout_kernel<<<1, 32>>>(d_B, d_results, variant);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_results, d_results, sizeof(h_results), cudaMemcpyDeviceToHost));
        
        // Reconstruct C matrix
        int32_t C[16][8] = {{0}};
        for (int lane = 0; lane < 32; lane++) {
            int groupID = lane / 4;
            int tid_in_group = lane % 4;
            int row0 = groupID;
            int row1 = groupID + 8;
            int col0 = tid_in_group * 2;
            int col1 = tid_in_group * 2 + 1;
            
            C[row0][col0] = h_results[lane * 4 + 0];
            C[row0][col1] = h_results[lane * 4 + 1];
            C[row1][col0] = h_results[lane * 4 + 2];
            C[row1][col1] = h_results[lane * 4 + 3];
        }
        
        int errors = 0;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 8; j++) {
                if (C[i][j] != 196) errors++;
            }
        }
        
        if (errors == 0) {
            printf("*** CORRECT! All values = 196 ***\n\n");
            printf("Winning B layout: variant %d\n", variant);
            // Print the C matrix for verification
            for (int i = 0; i < 16; i++) {
                printf("  Row %2d:", i);
                for (int j = 0; j < 8; j++) {
                    printf(" %4d", C[i][j]);
                }
                printf("\n");
            }
            break;
        } else {
            printf("Wrong: %d/128 errors. Sample: C[0][0]=%d, C[0][1]=%d\n\n", 
                   errors, C[0][0], C[0][1]);
        }
    }
    
    cudaFree(d_B);
    cudaFree(d_results);
}

// ============================================================================
// Test 3.5: Empirically discover A fragment layout
// ============================================================================
// Set A[i][k] = i*100 + k (unique per element), B = all 1s
// Then each output C[i][j] = sum_k(A[i][k] * 1) = sum_k(i*100 + k) = i*100*32 + sum(0..31) = i*3200 + 496
// By checking which thread produces which result, we can deduce the layout

__global__ void test_discover_a_layout_kernel(
    const int8_t* A,  // 16x32, A[i][k] = (i*4 + k/8) % 16 to fit in s4
    int32_t* per_thread_results  // 32 threads * 4 registers
) {
    const int lane = threadIdx.x % 32;
    
    // We'll try different loading strategies and see which one produces correct results
    // Strategy: load based on groupID/tid_in_group but store raw to analyze
    
    int groupID = lane / 4;
    int tid_in_group = lane % 4;
    int k_base = tid_in_group * 8;
    
    // Load A with our assumed layout
    uint32_t a0 = 0;
    for (int i = 0; i < 8; i++) {
        int8_t val = A[groupID * 32 + k_base + i];
        a0 |= ((uint32_t)(val & 0xF)) << (i * 4);
    }
    
    uint32_t a1 = 0;
    for (int i = 0; i < 8; i++) {
        int8_t val = A[(groupID + 8) * 32 + k_base + i];
        a1 |= ((uint32_t)(val & 0xF)) << (i * 4);
    }
    
    // B = all 1s
    uint32_t b0 = 0x11111111;
    
    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int32_t d0, d1, d2, d3;
    
    mma_m16n8k32_s4(d0, d1, d2, d3, a0, a1, b0, c0, c1, c2, c3);
    
    per_thread_results[lane * 4 + 0] = d0;
    per_thread_results[lane * 4 + 1] = d1;
    per_thread_results[lane * 4 + 2] = d2;
    per_thread_results[lane * 4 + 3] = d3;
}

void test_discover_a_layout() {
    printf("\n=== Test 3.5: Discover A fragment layout ===\n");
    
    int8_t h_A[16 * 32];
    
    // Fill A with pattern: A[i][k] = i (row index, clamped to s4)
    // Expected: C[i][j] = i * 32 (since B=1)
    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 32; k++) {
            h_A[i * 32 + k] = (int8_t)(i < 8 ? i : 7);  // clamp to s4
        }
    }
    
    int8_t* d_A;
    int32_t* d_results;
    int32_t h_results[32 * 4];
    
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(h_A)));
    CHECK_CUDA(cudaMalloc(&d_results, sizeof(h_results)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    
    test_discover_a_layout_kernel<<<1, 32>>>(d_A, d_results);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_results, d_results, sizeof(h_results), cudaMemcpyDeviceToHost));
    
    printf("With A[i][k] = min(i, 7), B[k][j] = 1:\n");
    printf("Expected: C[i][j] = min(i,7) * 32 for rows 0-7, = 7*32=224 for rows 8-15\n\n");
    
    // Reconstruct C matrix from thread outputs using known output layout
    int32_t C[16][8] = {{0}};
    for (int lane = 0; lane < 32; lane++) {
        int groupID = lane / 4;
        int tid_in_group = lane % 4;
        int row0 = groupID;
        int row1 = groupID + 8;
        int col0 = tid_in_group * 2;
        int col1 = tid_in_group * 2 + 1;
        
        C[row0][col0] = h_results[lane * 4 + 0];
        C[row0][col1] = h_results[lane * 4 + 1];
        C[row1][col0] = h_results[lane * 4 + 2];
        C[row1][col1] = h_results[lane * 4 + 3];
    }
    
    printf("Reconstructed C matrix:\n");
    for (int i = 0; i < 16; i++) {
        printf("  Row %2d:", i);
        for (int j = 0; j < 8; j++) {
            printf(" %4d", C[i][j]);
        }
        int expected = (i < 8 ? i : 7) * 32;
        printf("  (expected: %d)\n", expected);
    }
    
    // Now let's see what values each thread actually loaded
    printf("\nPer-thread raw outputs (to deduce which A elements each thread used):\n");
    for (int lane = 0; lane < 32; lane++) {
        printf("  Lane %2d (group=%d, tid=%d): d0=%4d d1=%4d d2=%4d d3=%4d\n",
               lane, lane/4, lane%4,
               h_results[lane*4+0], h_results[lane*4+1],
               h_results[lane*4+2], h_results[lane*4+3]);
    }
    
    cudaFree(d_A);
    cudaFree(d_results);
}

// ============================================================================
// Test 4: Full correctness test with random values
// ============================================================================
// 
// PTX m16n8k32 fragment layout (from NVIDIA PTX ISA documentation):
//
// A matrix (16x32, row-major):
//   Each thread holds 2 registers (a0, a1), each with 8 s4 values
//   Thread lane -> which elements:
//     a0: A[lane/4][((lane%4)*8) : ((lane%4)*8)+7] for rows 0-7 (repeated pattern)
//     a1: A[lane/4+8][((lane%4)*8) : ((lane%4)*8)+7] for rows 8-15
//   
//   Actually the layout is more complex. Let me derive from first principles.
//   The MMA needs 16*32 = 512 s4 values = 256 bytes = 64 uint32
//   32 threads, so each thread has 64/32 = 2 uint32 for A
//
// B matrix (32x8, col-major):
//   Each thread holds 1 register (b0) with 8 s4 values
//   32*8 = 256 s4 = 128 bytes = 32 uint32
//   32 threads -> 1 uint32 each
//
// For m16n8k32 with A row-major, B col-major:
//   A fragment: a0 covers K=[0,15], a1 covers K=[16,31] 
//   Within each register: packed by row groups
//
// Empirically determined correct mapping:

__global__ void test_mma_random_kernel(
    const int8_t* A,  // 16x32 row-major s4 (stored as s8 for convenience)
    const int8_t* B,  // 32x8 col-major s4
    int32_t* C        // 16x8 row-major s32
) {
    const int lane = threadIdx.x % 32;
    
    // Correct A fragment loading for m16n8k32 row.col
    // Based on PTX documentation and empirical testing:
    // 
    // For A (16x32 row-major), each thread loads 16 consecutive s4 values:
    // - a0: 8 s4 from first half of K (K=0..15)  
    // - a1: 8 s4 from second half of K (K=16..31)
    //
    // Thread mapping: 
    //   lane determines (row_pair, k_offset_within_half)
    //   lane/4 -> row (0-7 for a0, +8 for a1)
    //   lane%4 -> which 8 K values within each half (0-7, 8-15, but need to figure out exact)
    //
    // Actually, for m16n8k32.row.col the A matrix layout is:
    //   Threads are grouped into 8 groups of 4
    //   Group g (= lane/4) handles rows g and g+8
    //   Thread within group t (= lane%4) handles K positions
    //
    // Let me try a different interpretation based on the hardware layout:
    // The s4 MMA works on 32 K elements at once
    // a0 and a1 together should cover all 32 K elements for the thread's assigned rows
    
    int groupID = lane / 4;        // 0-7, determines row
    int tid_in_group = lane % 4;   // 0-3, determines K position
    
    // For A fragment in m16n8k32:
    // Thread (groupID, tid_in_group) loads:
    //   a0: 8 s4 values from row=groupID, K=[tid_in_group*8 : tid_in_group*8+7]
    //   a1: 8 s4 values from row=groupID+8, K=[tid_in_group*8 : tid_in_group*8+7]
    // 
    // But wait - that only covers K=[0:31] once across 4 threads, not 16x32
    // The issue is: each row needs ALL 32 K values, but 4 threads share a row
    // So the 4 threads in a group together load K=[0:31] for their shared row
    
    // Let's check: group 0 has lanes 0,1,2,3
    //   lane 0: K=[0:7], lane 1: K=[8:15], lane 2: K=[16:23], lane 3: K=[24:31]
    // Together they cover all 32 K for rows 0 and 8 - CORRECT!
    
    int k_base = tid_in_group * 8;  // 0, 8, 16, or 24
    
    // Pack a0: row = groupID, k = k_base..k_base+7
    uint32_t a0 = 0;
    for (int i = 0; i < 8; i++) {
        int8_t val = A[groupID * 32 + k_base + i];
        a0 |= ((uint32_t)(val & 0xF)) << (i * 4);
    }
    
    // Pack a1: row = groupID + 8, k = k_base..k_base+7  
    uint32_t a1 = 0;
    for (int i = 0; i < 8; i++) {
        int8_t val = A[(groupID + 8) * 32 + k_base + i];
        a1 |= ((uint32_t)(val & 0xF)) << (i * 4);
    }
    
    // For B fragment (32x8 col-major):
    // Correct layout discovered empirically (variant 2):
    //   b_col = lane / 4  (0-7, each column assigned to 4 consecutive lanes)
    //   b_k_base = (lane % 4) * 8  (0, 8, 16, 24 - which 8 K rows)
    // 
    // So lanes 0-3 handle column 0 at K=0-7, 8-15, 16-23, 24-31 respectively
    // Lanes 4-7 handle column 1, etc.
    
    int b_col = lane / 4;           // 0-7 (column index)
    int b_k_base = (lane % 4) * 8;  // 0, 8, 16, or 24
    
    uint32_t b0 = 0;
    for (int i = 0; i < 8; i++) {
        // Col-major: B[k][col] at index col*32 + k
        int8_t val = B[b_col * 32 + b_k_base + i];
        b0 |= ((uint32_t)(val & 0xF)) << (i * 4);
    }
    
    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int32_t d0, d1, d2, d3;
    
    mma_m16n8k32_s4(d0, d1, d2, d3, a0, a1, b0, c0, c1, c2, c3);
    
    // Output layout (verified correct in Test 3):
    // d0 -> C[groupID][tid_in_group * 2]
    // d1 -> C[groupID][tid_in_group * 2 + 1]
    // d2 -> C[groupID + 8][tid_in_group * 2]
    // d3 -> C[groupID + 8][tid_in_group * 2 + 1]
    
    int row0 = groupID;
    int row1 = groupID + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    
    C[row0 * 8 + col0] = d0;
    C[row0 * 8 + col1] = d1;
    C[row1 * 8 + col0] = d2;
    C[row1 * 8 + col1] = d3;
}

void test_mma_random() {
    printf("\n=== Test 4: Full correctness with random values ===\n");
    
    int8_t h_A[16 * 32];
    int8_t h_B[32 * 8];
    int32_t h_C_gpu[16 * 8];
    int32_t h_C_cpu[16 * 8];
    
    // Fill with small random values in s4 range [-8, 7]
    srand(42);
    for (int i = 0; i < 16 * 32; i++) {
        h_A[i] = (int8_t)((rand() % 16) - 8);
    }
    for (int i = 0; i < 32 * 8; i++) {
        h_B[i] = (int8_t)((rand() % 16) - 8);
    }
    
    // CPU reference computation
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            int32_t sum = 0;
            for (int k = 0; k < 32; k++) {
                // A is row-major, B is col-major
                sum += (int32_t)h_A[i * 32 + k] * (int32_t)h_B[j * 32 + k];
            }
            h_C_cpu[i * 8 + j] = sum;
        }
    }
    
    // GPU computation
    int8_t* d_A;
    int8_t* d_B;
    int32_t* d_C;
    
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(h_A)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(h_B)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(h_C_gpu)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeof(h_C_gpu)));
    
    test_mma_random_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, sizeof(h_C_gpu), cudaMemcpyDeviceToHost));
    
    // Compare
    printf("CPU reference C matrix:\n");
    for (int i = 0; i < 16; i++) {
        printf("  Row %2d:", i);
        for (int j = 0; j < 8; j++) {
            printf(" %5d", h_C_cpu[i * 8 + j]);
        }
        printf("\n");
    }
    
    printf("\nGPU MMA C matrix:\n");
    for (int i = 0; i < 16; i++) {
        printf("  Row %2d:", i);
        for (int j = 0; j < 8; j++) {
            printf(" %5d", h_C_gpu[i * 8 + j]);
        }
        printf("\n");
    }
    
    // Check for errors
    int errors = 0;
    for (int i = 0; i < 16 * 8; i++) {
        if (h_C_cpu[i] != h_C_gpu[i]) {
            errors++;
            if (errors <= 10) {
                printf("ERROR at [%d][%d]: CPU=%d, GPU=%d\n", 
                       i / 8, i % 8, h_C_cpu[i], h_C_gpu[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("\n*** PASS: All %d elements match! ***\n", 16 * 8);
    } else {
        printf("\n*** FAIL: %d errors out of %d elements ***\n", errors, 16 * 8);
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Test 5: Test with TCQ4-like block structure
// ============================================================================

struct block_tcq4_k32_test {
    uint8_t qs[128];    // 256 s4 packed
    uint16_t S;         // half as uint16
    uint16_t Z;         // half as uint16
    int8_t sc[8];       // per-group scales
    int8_t zc[8];       // per-group zeros
};

struct block_rrs_int4_test {
    uint8_t qs[128];    // 256 s4 packed
    float smooth_scale;
    int16_t sum_q[8];   // precomputed sums per group
};

__device__ float half_to_float(uint16_t h) {
    half hval;
    memcpy(&hval, &h, sizeof(half));
    return __half2float(hval);
}

// Test one TCQ4 block dot product
__global__ void test_tcq4_dot_kernel(
    const block_rrs_int4_test* A,
    const block_tcq4_k32_test* B,
    float* result
) {
    const int tid = threadIdx.x;
    
    // Each thread handles one of 8 groups
    if (tid >= 8) return;
    
    int g = tid;
    
    float smooth_scale = A->smooth_scale;
    float S = half_to_float(B->S);
    float Z = half_to_float(B->Z);
    float s_b = S * (float)B->sc[g];
    float z_b = Z * (float)B->zc[g];
    int16_t sum_qa = A->sum_q[g];
    
    // Compute dot product for this group (32 elements = 16 bytes)
    int32_t dot_qq = 0;
    
    for (int i = 0; i < 16; i++) {
        uint8_t a_byte = A->qs[g * 16 + i];
        uint8_t b_byte = B->qs[g * 16 + i];
        
        int8_t a0 = (a_byte & 0xF);
        a0 = (a0 >= 8) ? (a0 - 16) : a0;
        int8_t a1 = ((a_byte >> 4) & 0xF);
        a1 = (a1 >= 8) ? (a1 - 16) : a1;
        
        int8_t b0 = (b_byte & 0xF);
        b0 = (b0 >= 8) ? (b0 - 16) : b0;
        int8_t b1 = ((b_byte >> 4) & 0xF);
        b1 = (b1 >= 8) ? (b1 - 16) : b1;
        
        dot_qq += (int32_t)a0 * (int32_t)b0 + (int32_t)a1 * (int32_t)b1;
    }
    
    // Group contribution: smooth_scale * (s_b * dot_qq + z_b * sum_qa)
    float group_result = s_b * (float)dot_qq + z_b * (float)sum_qa;
    
    // Store per-group result
    result[g] = smooth_scale * group_result;
}

void test_tcq4_dot() {
    printf("\n=== Test 5: TCQ4 block dot product ===\n");
    
    block_rrs_int4_test h_A;
    block_tcq4_k32_test h_B;
    float h_result[8];
    
    // Initialize with known values
    h_A.smooth_scale = 1.5f;
    for (int i = 0; i < 128; i++) {
        h_A.qs[i] = 0x11;  // Two 1s per byte
    }
    for (int g = 0; g < 8; g++) {
        h_A.sum_q[g] = 32;  // 32 ones per group
    }
    
    half S_half = __float2half(2.0f);
    half Z_half = __float2half(0.5f);
    memcpy(&h_B.S, &S_half, sizeof(uint16_t));
    memcpy(&h_B.Z, &Z_half, sizeof(uint16_t));
    for (int i = 0; i < 128; i++) {
        h_B.qs[i] = 0x22;  // Two 2s per byte
    }
    for (int g = 0; g < 8; g++) {
        h_B.sc[g] = 1;
        h_B.zc[g] = 1;
    }
    
    // Expected per-group:
    // dot_qq = 32 * (1*2) = 64
    // s_b = 2.0 * 1 = 2.0
    // z_b = 0.5 * 1 = 0.5
    // group_result = 2.0 * 64 + 0.5 * 32 = 128 + 16 = 144
    // final = 1.5 * 144 = 216
    
    block_rrs_int4_test* d_A;
    block_tcq4_k32_test* d_B;
    float* d_result;
    
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(h_A)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(h_B)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(h_result)));
    
    CHECK_CUDA(cudaMemcpy(d_A, &h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    
    test_tcq4_dot_kernel<<<1, 8>>>(d_A, d_B, d_result);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_result, d_result, sizeof(h_result), cudaMemcpyDeviceToHost));
    
    printf("Per-group results:\n");
    float total = 0;
    for (int g = 0; g < 8; g++) {
        printf("  Group %d: %.2f (expected 216.00)\n", g, h_result[g]);
        total += h_result[g];
    }
    printf("Total: %.2f (expected 1728.00 = 8 * 216)\n", total);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("IMMA INT4 Tensor Core Micro Test Harness\n");
    printf("=========================================\n");
    
    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
        printf("ERROR: INT4 MMA requires SM 7.5+ (Turing or later)\n");
        return 1;
    }
    
    test_mma_ones();
    test_mma_row_index();
    test_mma_layout_discovery();
    test_discover_b_layout();
    test_discover_a_layout();
    test_mma_random();
    test_tcq4_dot();
    
    printf("\n=== All tests complete ===\n");
    return 0;
}