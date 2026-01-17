# RRS CUDA Benchmark - disabled due to build complexity
# The benchmark functionality is available directly in rrs.cu via:
# - ggml_cuda_rrs_benchmark()
# - ggml_cuda_rrs_print_benchmark()

# To benchmark RRS CUDA performance, call these functions from
# an existing CUDA-enabled test or add a test in the ggml-cuda tests.

message(STATUS "RRS benchmark example skipped (use ggml_cuda_rrs_benchmark() directly)")
