cmake_minimum_required(VERSION 3.12)

project(LearnTensorRT C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 添加可执行文件
add_executable(LearnTensorRT ${PROJECT_SOURCE_DIR}/cc/main.cpp)

file(GLOB_RECURSE SRCS "cc/*.cpp" "cc/*.h")

target_sources(LearnTensorRT
    PRIVATE
    ${SRCS}
)

# 添加包含目录
target_include_directories(LearnTensorRT
    PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/opencv-4.6.0/windows/include
    ${PROJECT_SOURCE_DIR}/3rdparty/TensorRT-8.4.1.5/windows/include
    ${PROJECT_SOURCE_DIR}/3rdparty/cuda-11.6/include
)

# 添加链接目录
target_link_directories(LearnTensorRT
    PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/opencv-4.6.0/windows/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/TensorRT-8.4.1.5/windows/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/cuda-11.6/lib/x64
)

# 链接库
target_link_libraries(LearnTensorRT
    nvinfer
)