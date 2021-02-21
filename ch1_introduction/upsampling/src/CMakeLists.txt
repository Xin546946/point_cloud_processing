add_library(bilateral_filter bilateral_filter.cpp)
target_link_libraries(bilateral_filter utils ${OpenCV_LIBS})