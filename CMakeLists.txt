cmake_minimum_required(VERSION 3.10)
project(nanodet_openvino)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)


# 添加可执行文件
add_executable(nanodet_openvino
    src/main.cpp
    src/Nanodet.cpp
    camera/hikvision_wrapper.cpp
)

# 包含目录
target_include_directories(nanodet_openvino PRIVATE
    include/
    camera/
    struct/
    /opt/MVS/include/           # 添加MVS头文件路径
)

# 添加hikvision SDK库路径
link_directories(/opt/MVS/bin)

# 链接库
target_link_libraries(nanodet_openvino PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
    MvCameraControl         # 添加海康MVS库
)

# 安装规则
install(TARGETS nanodet_openvino DESTINATION bin)
