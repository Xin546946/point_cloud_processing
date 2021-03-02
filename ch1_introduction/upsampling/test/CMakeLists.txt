add_executable(test_bilateral_filter test_bilateral_filter.cpp)
target_link_libraries(test_bilateral_filter bilateral_filter utils ${OpenCV_LIBS})

add_executable(test_upsampling test_upsampling.cpp)
target_link_libraries(test_upsampling bilateral_filter utils ${OpenCV_LIBS})

add_executable(test_upsamp_all_imgs test_upsamp_all_imgs.cpp)
target_link_libraries(test_upsamp_all_imgs bilateral_filter utils ${OpenCV_LIBS})