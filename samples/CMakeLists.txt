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

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_BINARY_DIR}/include
                    ${CMAKE_SOURCE_DIR}/external
                    ${TEMPEST_NDA_INCLUDE_DIRS})

ADD_DEFINITIONS("-DTempest_TESTS_DIR=\"${CMAKE_BINARY_DIR}\""
				"-DTEST_ASSETS_DIR=\"${CMAKE_SOURCE_DIR}/share/test_assets\""
                "-DROOT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")

# This is a convenience list. If you have something more elaborate in mind, please add your subdirectory
# with CMake file.

IF(NOT DISABLE_CUDA AND NOT UNIX)
    SET(CUDA_SAMPLES_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/object-paint.cu)
ENDIF()

SET(SAMPLES_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/cloth-simulation.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/debug-gizmos.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/virtual-gonioreflectometer.cc
    ${CUDA_SAMPLES_SRC}
    )

FOREACH(SAMPLE_SRC ${SAMPLES_SRC})
	IF(NOT EXISTS ${SAMPLE_SRC})
		MESSAGE(FATAL_ERROR "Test source does not exist: " ${SAMPLE_SRC})
	ENDIF()
	
	GET_FILENAME_COMPONENT(SAMPLE_NAME ${SAMPLE_SRC} NAME_WE)
	GET_FILENAME_COMPONENT(SAMPLE_DIR ${SAMPLE_SRC} DIRECTORY)
	
    IF(DISABLE_CUDA)
        ADD_EXECUTABLE(${SAMPLE_NAME} WIN32 ${SAMPLE_SRC})
    ELSE()
	    CUDA_ADD_EXECUTABLE(${SAMPLE_NAME} WIN32 ${SAMPLE_SRC})
    ENDIF()
	SET_PROPERTY(TARGET ${SAMPLE_NAME} PROPERTY FOLDER "Samples")
	SET_PROPERTY(TARGET ${SAMPLE_NAME} APPEND PROPERTY COMPILE_DEFINITIONS CURRENT_SOURCE_DIR="${SAMPLE_DIR}")
	TARGET_LINK_LIBRARIES(${SAMPLE_NAME} tempest-main tempest-static)
ENDFOREACH()
