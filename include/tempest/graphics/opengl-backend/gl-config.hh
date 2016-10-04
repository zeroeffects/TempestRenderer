/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2014 Zdravko Velinov
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*
*   The above copyright notice and this permission notice shall be included in
*   all copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*   THE SOFTWARE.
*/

#ifndef _TEMPEST_GL_CONFIG_HH_
#define _TEMPEST_GL_CONFIG_HH_

/////////////////////
// USER MODIFIABLE //
/////////////////////
#define TEMPEST_DISABLE_MDI_BINDLESS
#define TEMPEST_DISABLE_MDI
#define TEMPEST_DISABLE_TEXTURE_BINDLESS

/////////////////////////////////////
// DON'T MODIFY THESE ONES BY HAND //
/////////////////////////////////////
#define TEMPEST_RESOURCE_BUFFER 0
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
#   define TEMPEST_UBO_START 1
#endif

#if defined(TEMPEST_DISABLE_MDI) && !defined(TEMPEST_DISABLE_TEXTURE_BINDLESS)
#   define TEMPEST_GLOBALS_BUFFER 1
#   undef TEMPEST_UBO_START
#   define TEMPEST_UBO_START 2
#   define TEMPEST_SSBO_START 0
#else
#   define TEMPEST_GLOBALS_BUFFER 0
#   ifdef TEMPEST_DISABLE_MDI
#       define TEMPEST_SSBO_START 0
#       define TEMPEST_UBO_START 1
#   else
#       define TEMPEST_SSBO_START 1
#       ifndef TEMPEST_UBO_START
#           define TEMPEST_UBO_START 0
#       endif
#   endif
#endif

#endif