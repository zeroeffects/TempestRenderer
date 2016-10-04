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

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_BINARY_DIR}/include
                    ${CMAKE_SOURCE_DIR}/external
                    ${TEMPEST_NDA_INCLUDE_DIRS}
                   )

ADD_DEFINITIONS("-DCURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\""
                "-DTEST_ASSETS_DIR=\"${CMAKE_SOURCE_DIR}/share/test_assets\"")

SET(MESH_DEBUG_TOOL_SRC
	mesh-debug-tool.cc)

ADD_EXECUTABLE(mesh-debug-tool WIN32 ${MESH_DEBUG_TOOL_SRC})

SET_PROPERTY(TARGET mesh-debug-tool PROPERTY FOLDER "Tools")

TARGET_LINK_LIBRARIES(mesh-debug-tool tempest-main tempest-static)