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

FIND_PATH(EMBREE_INCLUDE_DIR NAMES "embree2/rtcore.h" PATHS "include")
FIND_LIBRARY(EMBREE_LIBRARIES NAMES embree PATHS "lib")

FIND_PATH(EMBREE_DYNAMIC_LIBRARY_PATH embree.dll PATHS "bin" "lib")
SET(EMBREE_DYNAMIC_LIBRARY ${EMBREE_DYNAMIC_LIBRARY_PATH}/embree.dll)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(EMBREE DEFAULT_MSG EMBREE_INCLUDE_DIR EMBREE_LIBRARIES)

MARK_AS_ADVANCED(EMBREE_INCLUDE_DIR EMBREE_LIBRARIES)