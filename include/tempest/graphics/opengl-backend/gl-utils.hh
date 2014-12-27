/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2010-2014 Zdravko Velinov
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

#ifndef _TEMPEST_GL_UTILS_HH_
#define _TEMPEST_GL_UTILS_HH_

#ifdef _WIN32
    #include <windows.h>
#endif

#include "tempest/graphics/opengl-backend/gl-library.hh"

#include "tempest/utils/logging.hh"
#include "tempest/graphics/rendering-definitions.hh"

namespace Tempest
{
string ConvertGLErrorToString(GLErrorCode err);

inline GLComparisonFunction TranslateComparisonFunction(ComparisonFunction compare_func)
{
    switch(compare_func)
    {
    default: TGE_ASSERT(false, "Unknown comparison function");
    case ComparisonFunction::Never: return GLComparisonFunction::GL_NEVER;
    case ComparisonFunction::Less: return GLComparisonFunction::GL_LESS;
    case ComparisonFunction::Equal: return GLComparisonFunction::GL_EQUAL;
    case ComparisonFunction::LessEqual: return GLComparisonFunction::GL_LEQUAL;
    case ComparisonFunction::Greater: return GLComparisonFunction::GL_GREATER;
    case ComparisonFunction::NotEqual: return GLComparisonFunction::GL_NOTEQUAL;
    case ComparisonFunction::GreaterEqual: return GLComparisonFunction::GL_GEQUAL;
    case ComparisonFunction::AlwaysPass: return GLComparisonFunction::GL_ALWAYS;
    }
}

#ifndef NDEBUG
inline void CheckOpenGL()
{
    auto opengl_err = glGetError();
    if(opengl_err != GLErrorCode::GL_NO_ERROR)
    {
        Log(LogLevel::Error, "OpenGL: error: ", ConvertGLErrorToString(opengl_err));
        TGE_ASSERT(opengl_err == GLErrorCode::GL_NO_ERROR, "An error has occurred while using OpenGL");
    }
}
#else
#define CheckOpenGL()
#endif
}

#endif // _TEMPEST_GL_UTILS_HH_