// TCQ4-K32 test runner
// Build: cmake --build build --target llama-tcq4-test
// Run: ./build/bin/llama-tcq4-test

#include <cstdio>

// Declare the test function from rrs.cu
extern "C" void ggml_cuda_tcq4_test(void);

int main() {
    printf("Running TCQ4-K32 tests...\n");
    ggml_cuda_tcq4_test();
    return 0;
}