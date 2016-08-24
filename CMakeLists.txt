cmake_minimum_required (VERSION 2.6)
project (StreamFilter)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -Wall -W -pedantic -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -W -pedantic -g")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)


FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(TBB REQUIRED)


include_directories ( include )
include_directories(${TBB_INCLUDE_DIRS})
add_definitions(${TBB_DEFINITIONS})

add_executable(StreamFilterApp src/main.cpp src/ErodeFilter.cpp  src/inputFilter.cpp src/outputFilter.cpp)
add_executable(sres src/single.cpp)

target_link_libraries(StreamFilterApp ${TBB_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(sres ${TBB_LIBRARIES} ${OpenCV_LIBS})
