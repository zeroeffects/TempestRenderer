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
					${FLEX_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${WINDOW_SYSTEM_INCLUDE_DIR})

SET(TEMPEST_UTILS_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/assert.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/file-system.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/interleave-vertices.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/library.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/logging.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/macros.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/memory.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/parse-command-line.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/patterns.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/system.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/testing.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/types.hh
   )

IF(WIN32)
	SET(TEMPEST_PLATFORM_UTILS_SRC utils/misc-w32.cc)
ENDIF()
   
SET(TEMPEST_UTILS_SRC
	${TEMPEST_PLATFORM_UTILS_SRC}
    utils/assert.cc
    utils/memory.cc
    utils/parse-command-line.cc
    utils/system.cc
    utils/logging.cc
    utils/library.cc
    utils/file-system.cc
    utils/interleave-vertices.cc
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

IF(UNIX)
    SET(TEMPEST_GL_PLATFORM_SPECIFIC_SRC graphics/opengl-backend/gl-window-x11.cc)
ELSEIF(WIN32)
	SET(TEMPEST_GL_PLATFORM_SPECIFIC_SRC
		graphics/opengl-backend/gl-window-w32.cc
		graphics/os-window-w32.cc)
ENDIF()
   
SET(TEMPEST_GRAPHICS_SRC
	${TEMPEST_GL_PLATFORM_SPECIFIC_SRC}
    graphics/opengl-backend/gl-library.cc
    graphics/opengl-backend/gl-backend.cc
    graphics/opengl-backend/gl-convenience.cc
    graphics/opengl-backend/gl-command-buffer.cc
    graphics/opengl-backend/gl-compiler.cc
    graphics/opengl-backend/gl-shader.cc
    graphics/opengl-backend/gl-buffer.cc
    graphics/opengl-backend/gl-input-layout.cc
    graphics/opengl-backend/gl-texture.cc
    graphics/rendering-convenience.cc
   )

SET(TEMPEST_GRAPHICS_INC
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-all.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-backend.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-buffer.hh
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-command-buffer.hh
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-compiler.hh
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-convenience.hh
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-input-layout.hh
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-input-layout.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-library.hh
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-shader.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-texture.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-utils.hh
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-window.hh
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/os-window.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/rendering-backend.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/rendering-definitions.hh
	${CMAKE_SOURCE_DIR}/include/tempest/graphics/texture.hh
  )

SET(TEMPEST_IMAGE_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/image/image.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/image/tga-image.hh
   )

SET(TEMPEST_IMAGE_SRC
    ${CMAKE_SOURCE_DIR}/src/image/image.cc
    ${CMAKE_SOURCE_DIR}/src/image/tga-image.cc
   )
   
IF(UNIX)   
	SET(TEMPEST_INPUT_SRC
		input/controller-linux.cc
	   )
ELSEIF(WIN32)
	SET(TEMPEST_INPUT_SRC
		input/controller-xinput.cc
	   )
ENDIF()

SET(TEMPEST_INPUT_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/input/controller.hh
   ) 

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shader)
   
BISON_TARGET(TempestParser "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader-parser.y"
             "${CMAKE_CURRENT_BINARY_DIR}/shader/shader-parser.cc"
             VERBOSE "${CMAKE_CURRENT_BINARY_DIR}/shader/shader-parser.output"
             COMPILE_FLAGS "--report=state")
FLEX_TARGET(TempestScanner "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader-lexer.l"
            "${CMAKE_CURRENT_BINARY_DIR}/shader/shader-lexer.cc"
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
    parser/driver-base.cpp
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

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-loader)
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-mtl-loader)
   
BISON_TARGET(ObjParser "${CMAKE_CURRENT_SOURCE_DIR}/mesh/obj-loader.y"
             "${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-loader/obj-loader-parser.cc"
             VERBOSE "${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-loader/obj-loader-parser.output"
             COMPILE_FLAGS "--report=state")
FLEX_TARGET(ObjScanner "${CMAKE_CURRENT_SOURCE_DIR}/mesh/obj-loader.l"
            "${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-loader/obj-loader-lexer.cc"
            COMPILE_FLAGS "-Pobj_loader_")
ADD_FLEX_BISON_DEPENDENCY(ObjScanner ObjParser)

BISON_TARGET(ObjMtlParser "${CMAKE_CURRENT_SOURCE_DIR}/mesh/obj-mtl-loader.y"
             "${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-mtl-loader/obj-mtl-loader-parser.cc"
             VERBOSE "${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-mtl-loader/obj-mtl-loader-parser.output"
             COMPILE_FLAGS "--report=state")
FLEX_TARGET(ObjMtlScanner "${CMAKE_CURRENT_SOURCE_DIR}/mesh/obj-mtl-loader.l"
            "${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-mtl-loader/obj-mtl-loader-lexer.cc"
            COMPILE_FLAGS "-Pobj_mtl_loader_")
ADD_FLEX_BISON_DEPENDENCY(ObjMtlScanner ObjMtlParser)

SET(TEMPEST_MESH_SRC
    mesh/obj-loader.cc
    mesh/obj-loader-driver.cc
    ${BISON_ObjParser_OUTPUTS}
    ${FLEX_ObjScanner_OUTPUTS}
    ${BISON_ObjMtlParser_OUTPUTS}
    ${FLEX_ObjMtlScanner_OUTPUTS}
   )

SET(TEMPEST_MESH_INC
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/obj-loader.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/obj-mtl-loader.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/obj-loader-driver.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/obj-mtl-loader-driver.hh
   )

SET(TEMPEST_LIBRARY_FILES
    ${TEMPEST_UTILS_INC}
    ${TEMPEST_UTILS_SRC}
    ${TEMPEST_LIB_INC}
    ${TEMPEST_GRAPHICS_SRC}
    ${TEMPEST_GRAPHICS_INC}
    ${TEMPEST_PARSER_SRC}
    ${TEMPEST_PARSER_INC}
    ${TEMPEST_IMAGE_SRC}
    ${TEMPEST_IMAGE_INC}
    ${TEMPEST_INPUT_SRC}
    ${TEMPEST_INPUT_INC}
    ${TEMPEST_MATH_SRC}
    ${TEMPEST_MATH_INC}
   )

IF(UNIX)
    SET(EXTRA_LIBS dl)
ENDIF()

ADD_LIBRARY(tempest-shader-tmp STATIC ${TEMPEST_SHADER_SRC} ${TEMPEST_SHADER_INC})
TARGET_INCLUDE_DIRECTORIES(tempest-shader-tmp PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/shader)

ADD_LIBRARY(tempest-mesh-tmp STATIC ${TEMPEST_MESH_SRC} ${TEMPEST_MESH_INC})
TARGET_INCLUDE_DIRECTORIES(tempest-mesh-tmp PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/mesh)

IF(UNIX)
	SET_TARGET_PROPERTIES(tempest-shader-tmp tempest-mesh-tmp PROPERTIES COMPILE_FLAGS "-fPIC")
ENDIF()

ADD_LIBRARY(tempest-static STATIC ${TEMPEST_LIBRARY_FILES})
TARGET_LINK_LIBRARIES(tempest-static ${EXTRA_LIBS} ${WINDOW_SYSTEM_LIBRARIES} tempest-shader-tmp tempest-mesh-tmp xxhash)

ADD_LIBRARY(tempest-shared SHARED ${TEMPEST_LIBRARY_FILES})
TARGET_LINK_LIBRARIES(tempest-shared ${EXTRA_LIBS} ${WINDOW_SYSTEM_LIBRARIES} tempest-shader-tmp tempest-mesh-tmp xxhash)


ADD_LIBRARY(tempest-main STATIC utils/tempest-main.cc)