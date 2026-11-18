cmake_minimum_required(VERSION 3.18)
project(segmosam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc video)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(segmosam
    cpp/src/tracker.cpp
    cpp/src/bindings.cpp
)

target_include_directories(segmosam PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(segmosam PRIVATE
    ${OpenCV_LIBS}
)

set_target_properties(segmosam PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)
