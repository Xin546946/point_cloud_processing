cmake_minimum_required(VERSION 3.10.2)
project(ISS)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -Wall -Wextra )
set(CMAKE_CXX_FLAGS  -o3)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV 3.2.0 REQUIRED)
find_package(PCL 1.11 REQUIRED)


include_directories(/usr/include/eigen3)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)
# include_directories(${CMAKE_SOURCE_DIR}/utils)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")


# add_subdirectory(utils)
add_subdirectory(src)
add_subdirectory(test)
