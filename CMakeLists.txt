cmake_minimum_required(VERSION 3.0.0)
project(neural_network VERSION 0.1.0)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Eigen_INCLUDE_DIRS} )

add_executable(neural_network main.cpp)

target_link_libraries(neural_network ${Eigen_LIBRARIES} ${OpenCV_LIBS})