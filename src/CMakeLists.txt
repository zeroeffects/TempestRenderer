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

IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/nda-build.cmake)
	INCLUDE(nda-build.cmake)
    SET(TEMPEST_NDA_CMAKE nda-build.cmake)
    SET(TEMPEST_NDA_APIS on PARENT_SCOPE)
    SET(TEMPEST_NDA_INCLUDE_DIRS ${TEMPEST_NDA_INCLUDE_DIRS} PARENT_SCOPE)
ENDIF()

ADD_DEFINITIONS(${PNG_DEFINITIONS} ${SIMD_FLAGS})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_BINARY_DIR}/include
                    ${CMAKE_SOURCE_DIR}/external
                    ${CMAKE_SOURCE_DIR}/external/libvpx
                    ${FLEX_INCLUDE_DIRS}
                    # ${PNG_INCLUDE_DIRS}
                    ${EMBREE_INCLUDE_DIR}
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${WINDOW_SYSTEM_INCLUDE_DIR}
                    ${TEMPEST_NDA_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/external/kissfft
                    ${CMAKE_SOURCE_DIR}/external/glslang
                    )

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
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/testing.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/display-image.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/refractive-indices.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/threads.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/video-encode.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/config.hh.in
	${CMAKE_SOURCE_DIR}/include/tempest/utils/viewer.hh
   )

IF(WIN32)
	SET(TEMPEST_PLATFORM_UTILS_SRC utils/misc-w32.cc)
ENDIF()
   
SET(TEMPEST_UTILS_SRC
	${TEMPEST_PLATFORM_UTILS_SRC}
    utils/assert.cc
    utils/memory.cc
    utils/parse-command-line.cc
    utils/logging.cc
    utils/library.cc
    utils/file-system.cc
    utils/threads.cc
    utils/interleave-vertices.cc
    utils/video-encode.cc
    utils/viewer.cc
   )

IF(UNIX)
    SET(TEMPEST_GL_PLATFORM_SPECIFIC_SRC
        graphics/opengl-backend/gl-window-x11.cc
        graphics/os-window-x11.cc)
ELSEIF(WIN32)
	SET(TEMPEST_GL_PLATFORM_SPECIFIC_SRC
		graphics/opengl-backend/gl-window-w32.cc
		graphics/os-window-w32.cc)
ENDIF()

SET(TEMPEST_GRAPHICS_GL_SRC
	${TEMPEST_GL_PLATFORM_SPECIFIC_SRC}
    graphics/opengl-backend/gl-library.cc
    graphics/opengl-backend/gl-backend.cc
    graphics/opengl-backend/gl-command-buffer.cc
    graphics/opengl-backend/gl-compiler.cc
    graphics/opengl-backend/gl-shader.cc
    graphics/opengl-backend/gl-buffer.cc
    graphics/opengl-backend/gl-input-layout.cc
    graphics/opengl-backend/gl-texture.cc
	graphics/opengl-backend/gl-state-object.cc
    graphics/opengl-backend/gl-storage.cc
    graphics/opengl-backend/gl-io-command-buffer.cc
    graphics/opengl-backend/gl-framebuffer.cc
    )

SET(TEMPEST_GRAPHICS_GL_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-all.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-config.hh
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
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-state-object.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-storage.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-io-command-buffer.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/opengl-backend/gl-framebuffer.hh
    )

SET(TEMPEST_GRAPHICS_MAIN_SRC
	${TEMPEST_NDA_GRAPHICS_SRC}

    graphics/equirectangular-map.cc
    graphics/rendering-convenience.cc
    graphics/rendering-definitions.cc
    graphics/texture.cc
    graphics/ray-tracing/ray-tracing.cc
    graphics/cube-map.cc
    graphics/scene-rasterizer.cc
    graphics/scene-rasterizer-system.cc
   )

SET(TEMPEST_GRAPHICS_SRC
    ${TEMPEST_GRAPHICS_MAIN_SRC}
    ${TEMPEST_GRAPHICS_GL_SRC}
   )

SET(TEMPEST_GRAPHICS_MAIN_INC
    ${TEMPEST_NDA_GRAPHICS_INC}
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/api-all.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/os-window.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/rendering-backend.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/rendering-convenience.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/rendering-definitions.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/state-object.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/texture.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/cube-map.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/shader.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/ray-tracing/ray-tracing.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/ray-tracing/ray-tracing-common.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/ray-tracing/ray-tracing-system.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/ray-tracing/illumination-models.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/ray-tracing/ggx-models-impl.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/ray-tracing/beckmann-models-impl.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/ray-tracing/microfacet-common.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/software-rasterizer.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/sampling-wrapper.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/custom-samplers.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/equirectangular-map.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/scene-rasterizer.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/graphics/scene-rasterizer-system.hh
   )

SET(TEMPEST_GRAPHICS_INC
    ${TEMPEST_GRAPHICS_MAIN_INC}
    ${TEMPEST_GRAPHICS_GL_INC}
    )


SET(TEMPEST_IMAGE_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/image/btf.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/image/image.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/image/image-utils.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/image/tga-image.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/image/png-image.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/image/exr-image.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/image/eps-draw.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/image/image-process.hh
   )

SET(TEMPEST_IMAGE_SRC
    ${CMAKE_SOURCE_DIR}/src/image/btf.cc
    ${CMAKE_SOURCE_DIR}/src/image/image.cc
    ${CMAKE_SOURCE_DIR}/src/image/image-utils.cc
    ${CMAKE_SOURCE_DIR}/src/image/tga-image.cc
    ${CMAKE_SOURCE_DIR}/src/image/png-image.cc
    ${CMAKE_SOURCE_DIR}/src/image/exr-image.cc
    ${CMAKE_SOURCE_DIR}/src/image/eps-draw.cc
   )

SET(TEMPEST_TEXTURE_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/texture/texture-table.hh
   )

SET(TEMPEST_TEXTURE_SRC
    ${CMAKE_SOURCE_DIR}/src/texture/texture-table.cc
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

SET(TEMPEST_VOLUME_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/volume/volume.hh
   )

SET(TEMPEST_VOLUME_SRC
    volume/volume.cc
   )

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shader)
   
FLEX_TARGET(TempestScanner "${CMAKE_CURRENT_SOURCE_DIR}/shader/shader-lexer.l"
            "${CMAKE_CURRENT_BINARY_DIR}/shader/shader-lexer.cc"
            COMPILE_FLAGS "-Pshader_")

SET(TEMPEST_SHADER_SRC
    shader/shader-lexer.l
    shader/dx-shader-generator.cc
    shader/gl-shader-generator.cc
    shader/shader-convert-common.cc
    shader/shader-common.cc
    shader/shader-driver.cc
    shader/shader-ast.cc
    shader/shader-parser.cc
    ${FLEX_TempestScanner_OUTPUTS}
   )

SET(TEMPEST_SHADER_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/dx-shader-generator.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/gl-shader-generator.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/shader-convert-common.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/shader-common.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/shader-driver.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/shader/shader-parser.hh
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
    math/dual-quaternion.cc
    math/shapes.cc
    math/spectrum.cc
    math/triangle.cc
    math/intersect.cc
    math/hdr.cc
    math/matrix-variadic.cc
   )

SET(TEMPEST_MATH_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/math/matrix4.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/vector4.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/vector3.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/vector2.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/shapes.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/sampling3.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/matrix3.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/matrix2.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/curves.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/spectrum.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/intersect.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/functions.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/windowing.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/matrix2.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/shape-split.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/numerical-methods.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/hdr.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/triangle.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/matrix-variadic.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/gabor-noise.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/point2.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/math/simple-basis.hh
   )

SET(TEMPEST_DEBUG_SRC
    debug/fps-counter.cc
   )

SET(TEMPEST_DEBUG_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/debug/fps-counter.hh
   )
   
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-loader)
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-mtl-loader)
   
FLEX_TARGET(ObjScanner "${CMAKE_CURRENT_SOURCE_DIR}/mesh/obj-loader.l"
            "${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-loader/obj-loader-lexer.cc"
            COMPILE_FLAGS "-Pobj_loader_")

FLEX_TARGET(ObjMtlScanner "${CMAKE_CURRENT_SOURCE_DIR}/mesh/obj-mtl-loader.l"
            "${CMAKE_CURRENT_BINARY_DIR}/mesh/obj-mtl-loader/obj-mtl-loader-lexer.cc"
            COMPILE_FLAGS "-Pobj_mtl_loader_")

SET(TEMPEST_MESH_SRC
    mesh/lbvh2.cc
    mesh/sslbvh2.cc
    mesh/essbvh2.cc
    mesh/obj-loader.l
    mesh/obj-loader.cc
    mesh/obj-loader-driver.cc
    mesh/obj-loader-parser.cc
    mesh/obj-mtl-loader.l
	mesh/obj-mtl-parser.cc
    mesh/octree.cc
    ${FLEX_ObjScanner_OUTPUTS}
    ${FLEX_ObjMtlScanner_OUTPUTS}
   )

SET(TEMPEST_MESH_INC
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/bvh-common.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/bvh-intersect-leaf.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/lbvh2.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/sslbvh2.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/essbvh2.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/octree.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/obj-loader.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/obj-loader-parser.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/obj-mtl-loader.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/obj-loader-driver.hh
   ${CMAKE_SOURCE_DIR}/include/tempest/mesh/obj-mtl-loader-driver.hh
   )
   
SET(TEMPEST_SYSTEM_INC
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/profiler.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/utils/system.hh
    )
    
SET(TEMPEST_SYSTEM_SRC
    utils/system.cc
    utils/profiler.cc
    )

IF(NOT DISABLE_CUDA)
    SET(TEMPEST_COMPUTE_SRC
	    compute/ray-tracing.cu
        compute/compute-definitions.cc
		compute/ray-tracing-cuda-system.cc
        compute/compute-texture.cc
        )

    SET(TEMPEST_COMPUTE_INC
        ${CMAKE_SOURCE_DIR}/include/tempest/compute/ray-tracing-cuda.hh
        ${CMAKE_SOURCE_DIR}/include/tempest/compute/ray-tracing-cuda-system.hh
        ${CMAKE_SOURCE_DIR}/include/tempest/compute/software-rasterizer-cuda.hh
        ${CMAKE_SOURCE_DIR}/include/tempest/compute/compute-definitions.hh
        ${CMAKE_SOURCE_DIR}/include/tempest/compute/compute-convenience.hh
        ${CMAKE_SOURCE_DIR}/include/tempest/compute/fitting-cuda.hh
        ${CMAKE_SOURCE_DIR}/include/tempest/compute/compute-texture.hh
        )
ENDIF()
   
SET(TEMPEST_LIBRARY_FILES
    ${TEMPEST_NDA_CMAKE}
    ${TEMPEST_UTILS_INC}
    ${TEMPEST_UTILS_SRC}
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
    ${TEMPEST_SHADER_SRC}
    ${TEMPEST_SHADER_INC}
    ${TEMPEST_MESH_SRC}
    ${TEMPEST_MESH_INC}
    ${TEMPEST_TEXTURE_SRC}
    ${TEMPEST_TEXTURE_INC} 
    ${TEMPEST_VOLUME_SRC}
    ${TEMPEST_VOLUME_INC}
    ${TEMPEST_DEBUG_SRC}
    ${TEMPEST_DEBUG_INC}
    ${TEMPEST_COMPUTE_SRC}
    ${TEMPEST_COMPUTE_INC}
    ${TEMPEST_SYSTEM_SRC}
    ${TEMPEST_SYSTEM_INC}
   )

SOURCE_GROUP("Utility functions" FILES ${TEMPEST_UTILS_INC}
									   ${TEMPEST_UTILS_SRC})
SOURCE_GROUP("Graphics" FILES ${TEMPEST_GRAPHICS_MAIN_SRC}
							  ${TEMPEST_GRAPHICS_MAIN_INC})
SOURCE_GROUP(Graphics\\OpenGL FILES ${TEMPEST_GRAPHICS_GL_SRC}
                                    ${TEMPEST_GRAPHICS_GL_INC})
SOURCE_GROUP("Compute" FILES ${TEMPEST_COMPUTE_SRC}
                             ${TEMPEST_COMPUTE_INC})
SOURCE_GROUP("File parser" FILES ${TEMPEST_PARSER_SRC}
								 ${TEMPEST_PARSER_INC})
SOURCE_GROUP("Image loader" FILES ${TEMPEST_IMAGE_SRC}
								  ${TEMPEST_IMAGE_INC})
SOURCE_GROUP("Input" FILES ${TEMPEST_INPUT_SRC}
						   ${TEMPEST_INPUT_INC})
SOURCE_GROUP("Math" FILES ${TEMPEST_MATH_SRC}
						  ${TEMPEST_MATH_INC})
SOURCE_GROUP("Shader parser" FILES ${TEMPEST_SHADER_SRC}
                                   ${TEMPEST_SHADER_INC})
SOURCE_GROUP("Mesh" FILES ${TEMPEST_MESH_SRC}
                          ${TEMPEST_MESH_INC}
                          ${TEMPEST_VOLUME_SRC}
                          ${TEMPEST_VOLUME_INC})

IF(UNIX)
    SET(EXTRA_LIBS dl)
ENDIF()

SET_PROPERTY(SOURCE ${TEMPEST_SHADER_SRC} ${TEMPEST_SHADER_INC} APPEND PROPERTY COMPILE_FLAGS
             "-I${CMAKE_CURRENT_BINARY_DIR}/shader")
             
SET_PROPERTY(SOURCE ${TEMPEST_MESH_SRC} ${TEMPEST_MESH_INC} APPEND PROPERTY COMPILE_FLAGS
             "-I${CMAKE_CURRENT_BINARY_DIR}/mesh")

SET(TEMPEST_DEP_LIBRARIES
    ${EXTRA_LIBS}
    ${TEMPEST_NDA_STATIC_LIBS}
    ${TEMPEST_NDA_LIBS}
    ${WINDOW_SYSTEM_LIBRARIES}
    # ${PNG_LIBRARIES}
    ${EMBREE_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${VPX_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    xxhash
    triangle
    faddeeva
    kissfft
    )

IF(DISABLE_CUDA)
    ADD_LIBRARY(tempest-static STATIC ${TEMPEST_LIBRARY_FILES})
ELSE()
    set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

    set(BUILD_SHARED_LIBS ON)

    CUDA_ADD_LIBRARY(tempest-static STATIC ${TEMPEST_LIBRARY_FILES})
ENDIF()

TARGET_LINK_LIBRARIES(tempest-static ${TEMPEST_DEP_LIBRARIES})

SET_PROPERTY(TARGET tempest-static PROPERTY FOLDER "Library")

#CUDA_ADD_LIBRARY(tempest-shared SHARED ${TEMPEST_LIBRARY_FILES})
#TARGET_LINK_LIBRARIES(tempest-shared ${TEMPEST_DEP_LIBRARIES})
#SET_PROPERTY(TARGET tempest-shared PROPERTY FOLDER "Library")

IF(UNIX)
# tempest-shared
    SET_TARGET_PROPERTIES(tempest-static PROPERTIES LINK_FLAGS "-Wl,--no-undefined")
ENDIF()

ADD_LIBRARY(tempest-main STATIC utils/tempest-main.cc)
SET_PROPERTY(TARGET tempest-main PROPERTY FOLDER "Library")

###################################################################################################

SET(TEMPEST_QT_WIDGETS_SRC
	qt/tempest-widget.cc
    qt/color-picker.cc)

SET(TEMPEST_QT_WIDGETS_INC
	${CMAKE_SOURCE_DIR}/include/tempest/qt/tempest-widget.hh
    ${CMAKE_SOURCE_DIR}/include/tempest/qt/color-picker.hh)

QT5_WRAP_CPP(TEMPEST_QT_WIDGETS_INC_MOC ${TEMPEST_QT_WIDGETS_INC})

ADD_LIBRARY(tempest-qt-widgets STATIC ${TEMPEST_QT_WIDGETS_SRC} ${TEMPEST_QT_WIDGETS_INC} ${TEMPEST_QT_WIDGETS_INC_MOC})
TARGET_LINK_LIBRARIES(tempest-qt-widgets Qt5::Widgets)
SET_PROPERTY(TARGET tempest-qt-widgets PROPERTY FOLDER "Library")
