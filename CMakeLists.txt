cmake_minimum_required(VERSION 3.10)
project(nanodet_openvino)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(yaml-cpp REQUIRED)

# 添加可执行文件
add_executable(nanodet_openvino
    src/main.cpp
    src/Nanodet.cpp
)

# 包含目录
target_include_directories(nanodet_openvino PRIVATE
    include/
)

# 链接库
target_link_libraries(nanodet_openvino PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
    yaml-cpp
)

# 安装规则
install(TARGETS nanodet_openvino DESTINATION bin)
