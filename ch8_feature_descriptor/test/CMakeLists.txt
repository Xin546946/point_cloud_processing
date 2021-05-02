add_executable(test_iss_pcl test_iss_pcl.cpp)
target_link_libraries(test_iss_pcl  ${PCL_LIBRARIES})

add_executable(test_my_iss test_my_iss.cpp)
target_link_libraries(test_my_iss iss ${PCL_LIBRARIES})