cmake_minimum_required(VERSION 3.12)
project(ml-cpp)

set(LIB srm_nn)
set(CMAKE_CXX_STANDARD 17)

# 查找软件包
find_package(OpenCV 4 REQUIRED)
find_package(Glog REQUIRED)

# 添加源文件
aux_source_directory(src SRC)
add_library(${LIB} SHARED ${SRC})

# 设置库的路径
target_compile_options(${LIB} PUBLIC -x objective-c++)
set(CoreML_LIBS "-framework AppKit -framework CoreML -framework Foundation -framework Vision -framework CoreVideo")

# 设置输出位置
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 设置头文件
target_include_directories(
  ${LIB}
  PUBLIC include
  PUBLIC ${OpenCV_INCLUDE_DIRS}
  PUBLIC ${GLOG_INCLUDE_DIRS}
)

# 设置链接库
target_link_libraries(
  ${LIB}
  PUBLIC ${CoreML_LIBS}
  PUBLIC ${OpenCV_LIBS}
  PUBLIC ${GLOG_LIBRARIES}
  PUBLIC glog::glog 
)

# 编译主程序
add_executable(main main.cpp)
target_link_libraries(main ${LIB})