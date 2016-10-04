#################################################################################
#   The MIT License
#   
#   Tempest Engine
#   Copyright (c) 2014 Zdravko Velinov
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

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

SET(CMAKE_AUTOMOC ON)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_BINARY_DIR}/include
                    ${CMAKE_SOURCE_DIR}/external
                    ${TEMPEST_NDA_INCLUDE_DIRS}
                   )

ADD_DEFINITIONS("-DCURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\""
                "-DTEST_ASSETS_DIR=\"${CMAKE_SOURCE_DIR}/share/test_assets\"")

SET(SHADING_EVALUATION_UI shading-evaluation.ui)

QT5_WRAP_UI(SHADING_EVALUATION_UI_H ${SHADING_EVALUATION_UI})

SET(SHADING_EVALUATION_SRC
    shading-evaluation.cc)

ADD_EXECUTABLE(shading-evaluation WIN32 ${SHADING_EVALUATION_SRC} ${SHADING_EVALUATION_UI} ${SHADING_EVALUATION_UI_H})
SET_PROPERTY(TARGET shading-evaluation PROPERTY FOLDER "Tools")

TARGET_LINK_LIBRARIES(shading-evaluation tempest-qt-widgets tempest-main tempest-static)