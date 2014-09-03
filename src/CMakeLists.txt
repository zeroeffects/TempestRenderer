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
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${WINDOW_SYSTEM_INCLUDE_DIR})

SET(TEMPEST_UTILS_SRC
    utils/assert.cc
    utils/memory.cc
    utils/parse-command-line.cc
    utils/system.cc
    utils/logging.cc
    utils/library.cc
    utils/file-system.cc
   )
   
SET(TEMPEST_LIB_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/assert.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/config.hh.in
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/memory.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/parse-command-line.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/patterns.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/system.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/macros.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/library.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/file-system.hh
   )

SET(TEMPEST_GRAPHICS_SRC
    graphics/opengl-backend/gl-library.cc
    graphics/opengl-backend/gl-backend.cc
    graphics/opengl-backend/gl-window-x11.cc
    graphics/opengl-backend/gl-convenience.cc
    graphics/opengl-backend/gl-command-buffer.cc
    graphics/opengl-backend/gl-compiler.cc
    graphics/opengl-backend/gl-shader.cc
    graphics/opengl-backend/gl-buffer.cc
    graphics/opengl-backend/gl-input-layout.cc
    graphics/rendering-convenience.cc
   )

SET(TEMPEST_GRAPHICS_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-library.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-backend.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-utils.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/rendering-definitions.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/rendering-backend.hh
   )

SET(TEMPEST_INPUT_SRC
    input/controller-linux.cc
   )

SET(TEMPEST_INPUT_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/input/controller.hh
   ) 

BISON_TARGET(TempestParser "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader-parser.y"
             "${CMAKE_CURRENT_BINARY_DIR}/shader-parser.cc"
             VERBOSE "${CMAKE_CURRENT_BINARY_DIR}/shader-parser.output"
             COMPILE_FLAGS "--report=state")
FLEX_TARGET(TempestScanner "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader-lexer.l"
            "${CMAKE_CURRENT_BINARY_DIR}/shader-lexer.cc"
            COMPILE_FLAGS "-Pshader_")
ADD_FLEX_BISON_DEPENDENCY(TempestScanner TempestParser)

SET(TEMPEST_SHADER_SRC
    shader/dx-shader-generator.cc
    shader/gl-shader-generator.cc
    shader/shader-common.cc
    shader/shader-driver.cc
    shader/shader-ast.cc
    ${BISON_TempestParser_OUTPUTS}
    ${FLEX_TempestScanner_OUTPUTS}
   )

SET(TEMPEST_SHADER_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/dx-shader-generator.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/gl-shader-generator.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/shader-common.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/shader-driver.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/shader-ast.hh
   )

SET(TEMPEST_PARSER_SRC
    parser/ast.cc
   ) 

SET(TEMPEST_PARSER_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/parser/ast.hh
   )

SET(TEMPEST_MATH_SRC
    math/matrix4.cc
    math/quaternion.cc
    math/dual-quaternion.cc
   )

SET(TEMPEST_MATH_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/math/matrix4.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/vector4.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/vector3.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/vector2.hh
   )

SET(TEMPEST_LIBRARY_FILES
    ${TEMPEST_UTILS_SRC}
    ${TEMPEST_LIB_INC}
    ${TEMPEST_GRAPHICS_SRC}
    ${TEMPEST_GRAPHICS_INC}
    ${TEMPEST_SHADER_SRC}
    ${TEMPEST_SHADER_INC}
    ${TEMPEST_PARSER_SRC}
    ${TEMPEST_PARSER_INC}
    ${TEMPEST_INPUT_SRC}
    ${TEMPEST_INPUT_INC}
    ${TEMPEST_MATH_SRC}
    ${TEMPEST_MATH_INC}
   )

IF(UNIX)
    SET(EXTRA_LIBS dl)
ENDIF()

ADD_LIBRARY(tempest-static STATIC ${TEMPEST_LIBRARY_FILES})
TARGET_LINK_LIBRARIES(tempest-static ${EXTRA_LIBS} ${WINDOW_SYSTEM_LIBRARIES})

ADD_LIBRARY(tempest-shared SHARED ${TEMPEST_LIBRARY_FILES})
TARGET_LINK_LIBRARIES(tempest-shared ${EXTRA_LIBS} ${WINDOW_SYSTEM_LIBRARIES})


ADD_LIBRARY(tempest-main STATIC utils/tempest-main.cc)