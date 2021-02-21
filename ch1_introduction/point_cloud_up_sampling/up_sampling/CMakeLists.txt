add_executable(up_sampling up_sampling.cpp)
target_link_libraries(up_sampling bilateral_filter ${OpenCV_LIBS})