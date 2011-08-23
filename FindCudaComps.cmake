###################################################################################
#
# Find CUDA and some CUDA libraries, such as 'cutil'
#
###################################################################################
# FindCUDA does not find our SDK, for some reason. We have an environment variable
# that points to it
set(CUDA_SDK_ROOT_DIR $ENV{NVSDKCUDA_ROOT})

# Find CUDA 
find_package(CUDA)
if(NOT CUDA_FOUND)
    set(DEFAULT FALSE)
    set(REASON "CUDA was not found.")
else(NOT CUDA_FOUND)
    set(DEFAULT TRUE)
    set(REASON)
    message(STATUS "CUDA found (include: ${CUDA_INCLUDE_DIRS}, lib: ${CUDA_LIBRARIES})")

    # Append configuration for arch 1.3 and 2.1 
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_13,code=sm_13")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_13,code=compute_13")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_21")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=compute_20")
endif(NOT CUDA_FOUND)

############################################################################
##### Find 'cutil' include file 
############################################################################
find_path(CUDA_CUT_INCLUDE_DIR
  cutil.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "C/common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
  )

# Now search system paths
find_path(CUDA_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")

############################################################################
##### Find 'cutil' library file 
############################################################################

# Define name of CUTIL based on build size
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil_x86_64)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil_x86_32)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

# Find library in custom paths
find_library(CUDA_CUT_LIBRARY
  NAMES cutil ${cuda_cutil_name}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  PATH_SUFFIXES "C/common/lib" "C/lib"
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
  )

# Now search system paths
find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})




