set(TARGET llama-tcq4-test)
add_executable(${TARGET} tcq4-test.cpp)
target_link_libraries(${TARGET} PRIVATE llama ggml common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
