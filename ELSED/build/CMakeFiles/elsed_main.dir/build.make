# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/build

# Include any dependencies generated for this target.
include CMakeFiles/elsed_main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/elsed_main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/elsed_main.dir/flags.make

CMakeFiles/elsed_main.dir/src/main.cpp.o: CMakeFiles/elsed_main.dir/flags.make
CMakeFiles/elsed_main.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/elsed_main.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elsed_main.dir/src/main.cpp.o -c /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/src/main.cpp

CMakeFiles/elsed_main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elsed_main.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/src/main.cpp > CMakeFiles/elsed_main.dir/src/main.cpp.i

CMakeFiles/elsed_main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elsed_main.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/src/main.cpp -o CMakeFiles/elsed_main.dir/src/main.cpp.s

# Object files for target elsed_main
elsed_main_OBJECTS = \
"CMakeFiles/elsed_main.dir/src/main.cpp.o"

# External object files for target elsed_main
elsed_main_EXTERNAL_OBJECTS =

elsed_main: CMakeFiles/elsed_main.dir/src/main.cpp.o
elsed_main: CMakeFiles/elsed_main.dir/build.make
elsed_main: libelsed.a
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
elsed_main: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
elsed_main: CMakeFiles/elsed_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable elsed_main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/elsed_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/elsed_main.dir/build: elsed_main

.PHONY : CMakeFiles/elsed_main.dir/build

CMakeFiles/elsed_main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/elsed_main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/elsed_main.dir/clean

CMakeFiles/elsed_main.dir/depend:
	cd /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/build /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/build /home/siddharth/viw_uvslam/src/VIW-Fusion_own/ELSED/build/CMakeFiles/elsed_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/elsed_main.dir/depend
