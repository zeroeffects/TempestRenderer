#################################################################################
#   The MIT License
#   
#   Tempest Engine
#   Copyright (c) 2016 Zdravko Velinov
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

IF(NOT DISABLE_CUDA)
    INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include
                        ${CMAKE_BINARY_DIR}/include
                        ${CMAKE_SOURCE_DIR}/external)

    SET(SVBRDF_VIEWER_SRC
	    svbrdf-viewer.cc
        svbrdf-viewer-cuda.cu)

    SET(SVBRDF_VIEWER_INC
        svbrdf-viewer.hh)

    CUDA_ADD_EXECUTABLE(svbrdf-viewer WIN32 ${SVBRDF_VIEWER_SRC} ${SVBRDF_VIEWER_INC})

    SET_PROPERTY(TARGET svbrdf-viewer PROPERTY FOLDER "Tools")

    TARGET_LINK_LIBRARIES(svbrdf-viewer tempest-main tempest-static)
ENDIF()