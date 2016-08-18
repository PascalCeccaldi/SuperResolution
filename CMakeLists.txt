cmake_minimum_required (VERSION 2.6)
project (StreamFilter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O5 -std=c++14")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

include_directories ( include )
add_subdirectory(src)

add_executable(StreamFilterApp src/main.cpp)
