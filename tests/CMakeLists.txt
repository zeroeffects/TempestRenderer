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

IF(NOT DISABLE_CUDA)
    SET(CUDA_TESTS_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/math/curve-fitting-test.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/pseudo-volume-performance-test.cu
	    ${CMAKE_CURRENT_SOURCE_DIR}/image/btf/btf-test-cuda.cu
	    ${CMAKE_CURRENT_SOURCE_DIR}/compute/ray-tracing-cuda-fill-test.cc
		${CMAKE_CURRENT_SOURCE_DIR}/graphics/interpolation/interpolation-test.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/graphics/interpolation/cube-map-interpolation-test.cu
        )
ENDIF()


# This is a convenience list. If you have something more elaborate in mind, please add your subdirectory
# with CMake file.
SET(TESTS_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/models/ashikhmin-shirley-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/copy-data/copy-data-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/draw-fps-counter/draw-fps-counter-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/draw-quad/draw-quad-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/draw-quad-texture/draw-quad-texture-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/draw-quad-texture-table/draw-quad-texture-table-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/mis/mis-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/object-file-viewer/object-file-viewer-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/tessellate-shapes/tessellate-shapes-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/window-init/window-init-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/yarn-lobe-diagram/yarn-lobe-diagram-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/input/controller/controller-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/numerical/brent-method/brent-method-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/shader/feature-check/feature-check-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/shader/parser/parser-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/utils/filesystem/fs-create-directory-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/utils/logging/logging-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/utils/patterns/packed-types-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/utils/patterns/scope-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/utils/solvers/solvers-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/volume/volume-read/volume-read-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/math/trigonometry-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/spectrum-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/misc/random-number-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark/sin-cos-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/sampling-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/procedural/velvet-weave-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/curves-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/models/ggx-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/matrix-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/rectangle-split-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/spring-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/software-rasterizer/rasterize-capsule-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/quaternion-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/models/sggx-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/models/brdf-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/numerical/integrator/integrator-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/video-encode/video-encode-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/video-encode/dyn-video-encode-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/image/image-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/fitting-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/image/btf/btf-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/delaunay-triangulation-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/intersect-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/intersect-line-circle-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/intersect-rect-circle-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/shapes-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/function-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh/bvh-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/mesh/mesh-render-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/ray-trace-quad/ray-trace-quad-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/graphics/texture/texture-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/triangle-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh/consistent-normals-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/draw-render-target/draw-render-target-test.cc
	${CMAKE_CURRENT_SOURCE_DIR}/mesh/octree-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/models/beckmann-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/texture/env-map-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/draw-text/draw-text-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/storage-ring/storage-ring-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/spherical-harmonics/spherical-harmonics-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics/cornell-box/cornell-box-test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/basis-test.cc
    ${CUDA_TESTS_SRC}
    )

FOREACH(TEST_SRC ${TESTS_SRC})
	IF(NOT EXISTS ${TEST_SRC})
		MESSAGE(FATAL_ERROR "Test source does not exist: " ${TEST_SRC})
	ENDIF()
	
	GET_FILENAME_COMPONENT(TEST_NAME ${TEST_SRC} NAME_WE)
	GET_FILENAME_COMPONENT(TEST_DIR ${TEST_SRC} DIRECTORY)
	
    IF(DISABLE_CUDA)
        ADD_EXECUTABLE(${TEST_NAME} WIN32 ${TEST_SRC})
    ELSE()
	    CUDA_ADD_EXECUTABLE(${TEST_NAME} WIN32 ${TEST_SRC})
    ENDIF()
	ADD_TEST(${TEST_NAME} ${TEST_NAME})
	SET_PROPERTY(TARGET ${TEST_NAME} PROPERTY FOLDER "Tests")
	SET_PROPERTY(TARGET ${TEST_NAME} APPEND PROPERTY COMPILE_DEFINITIONS CURRENT_SOURCE_DIR="${TEST_DIR}")
	TARGET_LINK_LIBRARIES(${TEST_NAME} tempest-main tempest-static ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUSOLVER_LIBRARIES})
ENDFOREACH()
