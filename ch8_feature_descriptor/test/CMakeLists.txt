add_executable(test_iss_pcl test_iss_pcl.cpp)
target_link_libraries(test_iss_pcl  ${PCL_LIBRARIES})

add_executable(test_my_iss test_my_iss.cpp)
target_link_libraries(test_my_iss fpfh iss ${PCL_LIBRARIES})

add_executable(test_spfh_resultset test_spfh_resultset.cpp)
target_link_libraries(test_spfh_resultset spfh)


add_executable(test_my_fpfh test_my_fpfh.cpp)
target_link_libraries(test_my_fpfh spfh fpfh ${PCL_LIBRARIES})