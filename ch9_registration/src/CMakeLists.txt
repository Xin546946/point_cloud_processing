add_library(iss iss.cpp)
target_link_libraries(iss ${PCL_LIBRARIES})

add_library(fpfh fpfh.cpp)
target_link_libraries(fpfh spfh ${PCL_LIBRARIES})

add_library(spfh spfh.cpp)
target_link_libraries(spfh ${PCL_LIBRARIES})

add_library(ransac ransac.cpp)
target_link_libraries(ransac ${PCL_LIBRARIES})