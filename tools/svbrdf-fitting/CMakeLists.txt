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
                    ${CUDA_INCLUDE_DIRS})

ADD_DEFINITIONS(-D_SCL_SECURE_NO_WARNINGS)

IF(NOT DISABLE_CUDA AND NOT UNIX)
    SET(TEMPEST_SVBRDF_FITTING_CUDA_SRC
        svbrdf-lsq-fitting.cu)
ENDIF()

SET(TEMPEST_SVBRDF_FITTING_INC
    svbrdf-fitting.hh
    svbrdf-lsq-fitting-internal.hh
    )

SET(TEMPEST_SVBRDF_FITTING_SRC
    svbrdf-fitting.cc
    svbrdf-lsq-fitting-setup.cc
    svbrdf-fitting-pipeline.cc
    svbrdf-display-single.cc
    svbrdf-display-distribution-map.cc
    ${TEMPEST_SVBRDF_FITTING_CUDA_SRC}
   )

IF(DISABLE_CUDA OR UNIX)
    ADD_EXECUTABLE(svbrdffit ${TEMPEST_SVBRDF_FITTING_INC} ${TEMPEST_SVBRDF_FITTING_SRC})
ELSE()
    CUDA_ADD_EXECUTABLE(svbrdffit ${TEMPEST_SVBRDF_FITTING_INC} ${TEMPEST_SVBRDF_FITTING_SRC})
ENDIF()

SET_PROPERTY(TARGET svbrdffit PROPERTY FOLDER "Tools")

TARGET_LINK_LIBRARIES(svbrdffit tempest-main tempest-static ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUSOLVER_LIBRARIES})

CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/svbrdf-fitting.job.in"
    "${CMAKE_CURRENT_BINARY_DIR}/svbrdf-fitting.job"
    )

IF(WIN32)
    FILE(COPY svbrdf-fitting.bat DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    FILE(COPY ${EMBREE_DYNAMIC_LIBRARY} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

    ADD_CUSTOM_COMMAND(TARGET svbrdffit POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:svbrdffit> ${CMAKE_CURRENT_BINARY_DIR}/
    )
ENDIF()