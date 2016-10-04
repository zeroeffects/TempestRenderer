#################################################################################
#   The MIT License
#   
#   Tempest Engine
#   Copyright (c) 2015 Zdravko Velinov
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

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_BINARY_DIR}/include
                    ${CMAKE_SOURCE_DIR}/external
                    ${TEMPEST_NDA_INCLUDE_DIRS}
                    ${Qt5Gui_PRIVATE_INCLUDE_DIRS}
                   )

ADD_DEFINITIONS("-DCURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\""
                "-DTEST_ASSETS_DIR=\"${CMAKE_SOURCE_DIR}/share/test_assets\"")

SET(MATERIAL_MANIPULATION_UI material-manipulation.ui)

QT5_WRAP_UI(MATERIAL_MANIPULATION_UI_H ${MATERIAL_MANIPULATION_UI})

SET(MATERIAL_MANIPULATION_SRC
    material-manipulation-main.cc
    )

SET(MATERIAL_MANIPULATION_INC
    material-manipulation.hh
    )

CUDA_COMPILE(MATERIAL_MANIPULATION_CUDA_OBJECTS material-manipulation.cu)  

ADD_EXECUTABLE(material-manipulation WIN32 ${MATERIAL_MANIPULATION_SRC} ${MATERIAL_MANIPULATION_INC} ${MATERIAL_MANIPULATION_UI} ${MATERIAL_MANIPULATION_UI_H} ${MATERIAL_MANIPULATION_CUDA_OBJECTS})
SET_PROPERTY(TARGET material-manipulation PROPERTY FOLDER "Tools")

TARGET_LINK_LIBRARIES(material-manipulation tempest-qt-widgets tempest-main tempest-static)
