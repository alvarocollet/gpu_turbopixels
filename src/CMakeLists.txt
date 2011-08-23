cmake_minimum_required( VERSION 2.8 )

# This CMakeLists is assumed to be loaded from the parent CMakeLists and inherits
# all the CUDA variables and ROS stuff

ADD_DEFINITIONS( -fPIC )


include_directories ( ${PROJECT_SOURCE_DIR}/include )
include_directories ( ${PROJECT_SOURCE_DIR}/src )
include_directories ( ${CUDA_CUT_INCLUDE_DIR} )

cuda_add_library( turbopix SHARED Turbopixels.cpp turbopix.cu reduction.cu recursiveGaussian.cu )
target_link_libraries ( turbopix ${CUDA_CUT_LIBRARY} )

###############################################################################
################ CUSTOM LINKING: OpenCV (C++) + CUDA ##########################
###############################################################################
# We don't need opencv to create the library, but we do for the executable
# We need custom linking for opencv for now, because we use cuda_add_executable

# HERE IS HOW TO DO IT: (No, find_package(OpenCV) does not work for us.

rosbuild_find_ros_package( opencv2 )
find_library( CVLIB NAMES cv.so HINTS ${opencv2_PACKAGE_PATH}/lib )
include_directories ( ${opencv2_PACKAGE_PATH}/opencv/include )
link_directories ( ${opencv2_PACKAGE_PATH}/lib )
cuda_add_executable( example_tpix example.cpp )
target_link_libraries ( example_tpix ${CVLIB} ${CUDA_CUT_LIBRARY} turbopix )

#target_link_libraries ( turbopix cv ${CUDA_CUT_LIBRARY})
#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(extract_3d_features src/extract_3d_features.cpp)
#target_link_libraries(example ${PROJECT_NAME})
