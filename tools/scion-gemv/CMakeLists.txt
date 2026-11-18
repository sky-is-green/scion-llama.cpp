set(TARGET scion-gemv-bench)

add_executable(${TARGET} gemv-bench.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
