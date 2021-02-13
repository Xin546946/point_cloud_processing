add_executable(test_pca test_pca.cpp)
target_link_libraries(test_pca pca ${OpenCV_LIBS})