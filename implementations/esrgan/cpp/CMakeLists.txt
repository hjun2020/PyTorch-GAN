cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})

link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_CFLAGS_OTHER})
link_directories(/usr/local/lib)


add_executable(esrgan esrgan.cpp)
target_link_libraries(esrgan "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
set_property(TARGET esrgan PROPERTY CXX_STANDARD 14)