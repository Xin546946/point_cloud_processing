add_executable(kdt kdt.cpp)
target_link_libraries(kdt ${OpenCV_LIBS})
target_link_libraries(kdt ${PCL_LIBRARIES})