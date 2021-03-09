add_executable(test_gkdtree test_gkdtree.cpp)
target_link_libraries(test_gkdtree gkdtree utils)

add_executable(test_pcl test_pcl.cpp)
target_link_libraries(test_pcl ${PCL_LIBRARIES})




