// TCQ4-K32 test runner
// Build: cmake --build build --target test-tcq4
// Run: ./build/bin/test-tcq4

#include <cstdio>

// Declare the test function from rrs.cu
extern "C" void ggml_cuda_tcq4_test(void);

int main() {
    printf("Running TCQ4-K32 tests...\n");
    ggml_cuda_tcq4_test();
    return 0;
}