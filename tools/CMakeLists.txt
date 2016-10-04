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

ADD_SUBDIRECTORY(convert-volume)
ADD_SUBDIRECTORY(shader-compiler)
ADD_SUBDIRECTORY(shading-evaluation)
ADD_SUBDIRECTORY(btf-viewer)
ADD_SUBDIRECTORY(svbrdf-viewer)
ADD_SUBDIRECTORY(btf-slice)
ADD_SUBDIRECTORY(btf-extract-normal-map)
ADD_SUBDIRECTORY(exr-viewer)
ADD_SUBDIRECTORY(quaternion-texture-tool)
ADD_SUBDIRECTORY(mesh-debug-tool)
ADD_SUBDIRECTORY(svbrdf-postprocess)
ADD_SUBDIRECTORY(svbrdf-fitting)
ADD_SUBDIRECTORY(btf-to-ambient-occlusion)
ADD_SUBDIRECTORY(svbrdf-error-tool)
ADD_SUBDIRECTORY(color-code-tool)
ADD_SUBDIRECTORY(image-combine)
ADD_SUBDIRECTORY(image-analyze)
ADD_SUBDIRECTORY(image-convert)
ADD_SUBDIRECTORY(texture-generate)
ADD_SUBDIRECTORY(image-histogram)
IF(NOT DISABLE_CUDA)
    ADD_SUBDIRECTORY(material-manipulation)
ENDIF()
ADD_SUBDIRECTORY(image-to-movie)
ADD_SUBDIRECTORY(env-map-tool)
ADD_SUBDIRECTORY(btf-reflectance-slice)
ADD_SUBDIRECTORY(btf-lobe-diagram-tool)
ADD_SUBDIRECTORY(fabrics-tester)