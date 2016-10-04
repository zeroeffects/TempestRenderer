#################################################################################
#   The MIT License
#   
#   Tempest Engine
#   Copyright (c) 2010-2016 Zdravko Velinov
#   
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
##################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE(CheckCXXCompilerFlag)

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

PROJECT(TempestVMV2016Revision)

CMAKE_POLICY(SET CMP0020 OLD)

SET(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo CACHE TYPE INTERNAL FORCE)

ADD_DEFINITIONS("-DSOURCE_SHADING_DIR=\"${CMAKE_SOURCE_DIR}/shading\"")

IF(APPLE)
    SET(EXTRA_DEFS "-DAPPLE")
ELSEIF(UNIX)
    SET(EXTRA_DEFS "-DLINUX -std=c++11")
    FIND_PACKAGE(X11 REQUIRED)
    SET(WINDOW_SYSTEM_INCLUDE_DIR ${X11_INCLUDE_DIR})
    SET(WINDOW_SYSTEM_LIBRARIES ${X11_LIBRARIES})
ENDIF()

IF(NOT DISABLE_CUDA)
    FIND_PACKAGE(CUDA REQUIRED)

    GET_FILENAME_COMPONENT(CUDA_LIBRARY_DIR "${CUDA_CUBLAS_LIBRARIES}" DIRECTORY)

    FIND_LIBRARY(CUDA_CUSOLVER_LIBRARIES cusolver PATHS ${CUDA_LIBRARY_DIR})
    IF(NOT CUDA_CUSOLVER_LIBRARIES)
        MESSAGE(FATAL_ERROR "Failed to find cuSOLVER")
    ENDIF()
ENDIF()
# FIND_PACKAGE(PNG REQUIRED)
FIND_PACKAGE(FLEX REQUIRED)
FIND_PACKAGE(Embree REQUIRED)
FIND_PACKAGE(Threads)

IF(WIN32)
    FIND_LIBRARY(VPX_LIBRARY vpxmt PATHS ${CMAKE_SOURCE_DIR}/external/libvpx/x64/Release)
ELSEIF(UNIX)
    SET(VPX_LIBRARY ${CMAKE_SOURCE_DIR}/external/libvpx/x64/Release/libvpx.a)
    MESSAGE(${VPX_LIBRARY})
ELSE()
    MESSAGE(FATAL_ERROR "Unsupported platform")
ENDIF()

IF(MSVC)
	ADD_DEFINITIONS("/DNOMINMAX /D_SCL_SECURE_NO_WARNINGS /D_CRT_SECURE_NO_WARNINGS /arch:AVX")
    SET(HAS_SSE on)
ELSE(MSVC)
    CHECK_CXX_COMPILER_FLAG(-mavx HAS_SSE)
    IF(HAS_SSE)
        SET(SIMD_FLAGS "-msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -mavx")
    ELSE(HAS_SSE)
        CHECK_CXX_COMPILER_FLAG(-mfpu=neon HAS_ARM_NEON)
        IF(HAS_ARM_NEON)
            SET(SIMD_FLAGS -mfpu=neon)
        ELSE(HAS_ARM_NEON)
            MESSAGE(WARNING "No supported SIMD instruction set has been found.")
        ENDIF(HAS_ARM_NEON)
    ENDIF(HAS_SSE)
ENDIF(MSVC)

ADD_DEFINITIONS(${EXTRA_DEFS} ${SIMD_FLAGS})

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50 ${EXTRA_DEFS})

SET(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -g -O0 -G)

FIND_PACKAGE(Qt5Widgets)

ADD_SUBDIRECTORY(include)
ADD_SUBDIRECTORY(share)
ADD_SUBDIRECTORY(src)
IF(NOT DISABLE_TESTS)
	ADD_SUBDIRECTORY(tests)
ENDIF()
ADD_SUBDIRECTORY(samples)
ADD_SUBDIRECTORY(tools)
ADD_SUBDIRECTORY(external/xxhash)
ADD_SUBDIRECTORY(external/triangle)
ADD_SUBDIRECTORY(external/faddeeva)
ADD_SUBDIRECTORY(external/kissfft)

CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/tempest/utils/config.hh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/tempest/utils/config.hh"
    )
