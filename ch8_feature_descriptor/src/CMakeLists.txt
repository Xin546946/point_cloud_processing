add_library(iss iss.cpp)
target_link_libraries(iss  ${PCL_LIBRARIES})

add_library(fpfh fpfh.cpp)
target_link_libraries(fpfh  ${PCL_LIBRARIES})
