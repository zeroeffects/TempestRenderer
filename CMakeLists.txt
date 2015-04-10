#################################################################################
#   The MIT License
#   
#   Tempest Engine
#   Copyright (c) 2010-2014 Zdravko Velinov
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

PROJECT(Tempest)

SET(CMAKE_VERBOSE_MAKEFILE on)

FIND_PACKAGE(PNG REQUIRED)
FIND_PACKAGE(BISON REQUIRED)
FIND_PACKAGE(FLEX REQUIRED)

IF(APPLE)
    ADD_DEFINITIONS("-DAPPLE")
ELSEIF(UNIX)
    ADD_DEFINITIONS("-DLINUX -std=c++11")
    FIND_PACKAGE(X11 REQUIRED)
    SET(WINDOW_SYSTEM_INCLUDE_DIR ${X11_INCLUDE_DIR})
    SET(WINDOW_SYSTEM_LIBRARIES ${X11_LIBRARIES})
ENDIF()

IF(MSVC)
	ADD_DEFINITIONS("/DNOMINMAX /D_SCL_SECURE_NO_WARNINGS")
    SET(SIMD_FLAGS "/arch:SSE")
    SET(HAS_SSE on)
ELSE(MSVC)
    CHECK_CXX_COMPILER_FLAG(-msse HAS_SSE)
    IF(HAS_SSE)
        SET(SIMD_FLAGS "-msse")
    ELSE(HAS_SSE)
        CHECK_CXX_COMPILER_FLAG(-mfpu=neon HAS_ARM_NEON)
        IF(HAS_ARM_NEON)
            SET(SIMD_FLAGS -mfpu=neon)
        ELSE(HAS_ARM_NEON)
            MESSAGE(WARNING "No supported SIMD instruction set has been found.")
        ENDIF(HAS_ARM_NEON)
    ENDIF(HAS_SSE)
ENDIF(MSVC)

FIND_PACKAGE(Qt5Widgets)
FIND_PACKAGE(Qt5Declarative)

ADD_SUBDIRECTORY(include)
ADD_SUBDIRECTORY(share)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(tests)
ADD_SUBDIRECTORY(tools)
ADD_SUBDIRECTORY(external/xxhash)

CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/tempest/utils/config.hh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/tempest/utils/config.hh"
    )