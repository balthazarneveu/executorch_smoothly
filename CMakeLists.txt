cmake_minimum_required(VERSION 3.19)
project(webcam_executorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set options for executorch build.
option(EXECUTORCH_ENABLE_LOGGING "" ON)
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON) # Build with XNNPACK backend

# Include the executorch subdirectory.
add_subdirectory(
    /Data/code/executorch
    /Data/code/executorch/third-party
)

# Find OpenCV package.
find_package(OpenCV REQUIRED)

# Add the executable target.
add_executable(webcam_executorch main.cpp)

# Link libraries to the target.
target_link_libraries(
  webcam_executorch
  PRIVATE executorch
          extension_module_static # Provides the Module class
          extension_tensor # Provides the TensorPtr class
          optimized_native_cpu_ops_lib # Provides baseline cross-platform kernels
          xnnpack_backend # Provides the XNNPACK CPU acceleration backend
          ${OpenCV_LIBS} # Link OpenCV libraries
)

# Include OpenCV headers.
target_include_directories(
  webcam_executorch
  PRIVATE ${OpenCV_INCLUDE_DIRS}
)
