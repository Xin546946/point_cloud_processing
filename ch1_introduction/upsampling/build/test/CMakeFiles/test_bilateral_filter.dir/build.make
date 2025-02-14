# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kit/point_cloud_processing/ch1_introduction/upsampling

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kit/point_cloud_processing/ch1_introduction/upsampling/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test_bilateral_filter.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_bilateral_filter.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_bilateral_filter.dir/flags.make

test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o: test/CMakeFiles/test_bilateral_filter.dir/flags.make
test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o: ../test/test_bilateral_filter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kit/point_cloud_processing/ch1_introduction/upsampling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o"
	cd /home/kit/point_cloud_processing/ch1_introduction/upsampling/build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o -c /home/kit/point_cloud_processing/ch1_introduction/upsampling/test/test_bilateral_filter.cpp

test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.i"
	cd /home/kit/point_cloud_processing/ch1_introduction/upsampling/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kit/point_cloud_processing/ch1_introduction/upsampling/test/test_bilateral_filter.cpp > CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.i

test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.s"
	cd /home/kit/point_cloud_processing/ch1_introduction/upsampling/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kit/point_cloud_processing/ch1_introduction/upsampling/test/test_bilateral_filter.cpp -o CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.s

test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o.requires:

.PHONY : test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o.requires

test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o.provides: test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/test_bilateral_filter.dir/build.make test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o.provides.build
.PHONY : test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o.provides

test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o.provides.build: test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o


# Object files for target test_bilateral_filter
test_bilateral_filter_OBJECTS = \
"CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o"

# External object files for target test_bilateral_filter
test_bilateral_filter_EXTERNAL_OBJECTS =

test/test_bilateral_filter: test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o
test/test_bilateral_filter: test/CMakeFiles/test_bilateral_filter.dir/build.make
test/test_bilateral_filter: src/libbilateral_filter.a
test/test_bilateral_filter: utils/libutils.a
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
test/test_bilateral_filter: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
test/test_bilateral_filter: test/CMakeFiles/test_bilateral_filter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kit/point_cloud_processing/ch1_introduction/upsampling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_bilateral_filter"
	cd /home/kit/point_cloud_processing/ch1_introduction/upsampling/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_bilateral_filter.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_bilateral_filter.dir/build: test/test_bilateral_filter

.PHONY : test/CMakeFiles/test_bilateral_filter.dir/build

test/CMakeFiles/test_bilateral_filter.dir/requires: test/CMakeFiles/test_bilateral_filter.dir/test_bilateral_filter.cpp.o.requires

.PHONY : test/CMakeFiles/test_bilateral_filter.dir/requires

test/CMakeFiles/test_bilateral_filter.dir/clean:
	cd /home/kit/point_cloud_processing/ch1_introduction/upsampling/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_bilateral_filter.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_bilateral_filter.dir/clean

test/CMakeFiles/test_bilateral_filter.dir/depend:
	cd /home/kit/point_cloud_processing/ch1_introduction/upsampling/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kit/point_cloud_processing/ch1_introduction/upsampling /home/kit/point_cloud_processing/ch1_introduction/upsampling/test /home/kit/point_cloud_processing/ch1_introduction/upsampling/build /home/kit/point_cloud_processing/ch1_introduction/upsampling/build/test /home/kit/point_cloud_processing/ch1_introduction/upsampling/build/test/CMakeFiles/test_bilateral_filter.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_bilateral_filter.dir/depend

