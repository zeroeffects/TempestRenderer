/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2009 2010 2011 2012 Zdravko Velinov
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


#define DECLARE_GL_FUNCTION(return_type, name, ...) typedef return_type (APIENTRY* PFN##name##PROC)(__VA_ARGS__); \
                                                    PFN##name##PROC name = nullptr;
#define DECLARE_GL_FUNCTION_OPTIONAL(caps, return_type, name, ...) typedef return_type (APIENTRY* PFN##name##PROC)(__VA_ARGS__); \
                                                                   PFN##name##PROC name = nullptr;
#define DECLARE_SYS_FUNCTION(return_type, name, ...) typedef return_type (APIENTRY* PFN##name##PROC)(__VA_ARGS__); \
                                                     PFN##name##PROC name = nullptr;
#define DECLARE_SYS_GL_FUNCTION(return_type, name, ...) typedef return_type (APIENTRY* PFN##name##PROC)(__VA_ARGS__); \
                                                        PFN##name##PROC name = nullptr;
#include "tempest/graphics/opengl-backend/gl-library.hh"
#undef DECLARE_GL_FUNCTION
#undef DECLARE_GL_FUNCTION_OPTIONAL
#undef DECLARE_SYS_FUNCTION
#undef DECLARE_SYS_GL_FUNCTION

namespace Tempest
{
#ifdef _WIN32
#   define GL_LIB_NAME "opengl32.dll"
#else
#   define GL_LIB_NAME "libGL.so"
#endif

#ifdef _WIN32
    PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB = nullptr;
    PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB = nullptr;
#endif

static uint64 GLCaps = ~0ULL;

bool IsGLCapabilitySupported(uint64 caps)
{
    return (GLCaps & caps) == caps;
}

string ConvertGLErrorToString(GLErrorCode err)
{
    switch(err)
    {
    case GLErrorCode::GL_NO_ERROR:
        return "no error"; break;
    case GLErrorCode::GL_INVALID_ENUM:
        return "invalid enumerated argument"; break;
    case GLErrorCode::GL_INVALID_VALUE:
        return "invalid value"; break;
    case GLErrorCode::GL_INVALID_OPERATION:
        return "invalid operation"; break;
    case GLErrorCode::GL_INVALID_FRAMEBUFFER_OPERATION:
        return "framebuffer object is incomplete";
    case GLErrorCode::GL_OUT_OF_MEMORY:
        return "out of memory"; break;
    default:
        break;
    }
    std::stringstream ss;
    ss << std::hex << (int)err;
    return ss.str();
}

#ifdef _WIN32
// Because f... you person that has invented this clunky API
    static HGLRC s_RC = nullptr;
#endif

GLLibrary::~GLLibrary()
{
#ifdef _WIN32
    if(s_RC)
    {
        wglDeleteContext(s_RC);
        s_RC = nullptr;
    }
#endif
}

#ifdef _WIN32
// Imagine that, you need regular context to query the extension.
bool InitDummyContext(HDC hDC)
{
    if(!s_RC)
    {
        PIXELFORMATDESCRIPTOR pfd =
        {
            sizeof(PIXELFORMATDESCRIPTOR),
            1,
            PFD_DRAW_TO_WINDOW |
            PFD_SUPPORT_OPENGL |
            PFD_DOUBLEBUFFER,
            PFD_TYPE_RGBA,
            24,
            0, 0, 0, 0, 0, 0,
            0,
            0,
            0,
            0, 0, 0, 0,
            24,
            8,
            0,
            PFD_MAIN_PLANE,
            0,
            0, 0, 0
        };

        int iPixelFormat = ChoosePixelFormat(hDC, &pfd);
        SetPixelFormat(hDC, iPixelFormat, &pfd);

        s_RC = wglCreateContext(hDC);
        TGE_ASSERT(s_RC, "Expecting valid context");
        if(!s_RC)
        {
            return false;
        }
        wglMakeCurrent(hDC, s_RC);

        wglCreateContextAttribsARB = reinterpret_cast<decltype(wglCreateContextAttribsARB)>(GL_GET_PROC_ADDRESS("wglCreateContextAttribsARB"));
        wglChoosePixelFormatARB = reinterpret_cast<decltype(wglChoosePixelFormatARB)>(GL_GET_PROC_ADDRESS("wglChoosePixelFormatARB"));
    }
    return true;
}

BOOL w32hackChoosePixelFormat(HDC hDC, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats)
{
    if(!InitDummyContext(hDC) || !wglChoosePixelFormatARB)
        return FALSE;
    return wglChoosePixelFormatARB(hDC, piAttribIList, pfAttribFList, nMaxFormats, piFormats, nNumFormats);
}

HGLRC w32hackCreateContextAttribs(HDC hDC, HGLRC hShareContext, const int *attribList)
{
    if(!InitDummyContext(hDC) || !wglCreateContextAttribsARB)
        return nullptr;
    return wglCreateContextAttribsARB(hDC, hShareContext, attribList);
}
#endif

bool GLLibrary::initDeviceContextLibrary()
{
    if(m_GLLib.loaded())
        return true;

    if(!m_GLLib.load(GL_LIB_NAME))
        return false;

    #define TEMPEST_EXTRACT_FUNCTIONS
    #define DECLARE_GL_FUNCTION(return_type, name, ...)
    #define DECLARE_GL_FUNCTION_OPTIONAL(caps, return_type, name, ...)
    #define DECLARE_SYS_FUNCTION(return_type, name, ...) GL_LIB_LOAD_FUNCTION(name)
    #define DECLARE_SYS_GL_FUNCTION(return_type, name, ...) GL_LOAD_FUNCTION(name)
    #include "tempest/graphics/opengl-backend/gl-library.hh"
    #undef TEMPEST_EXTRACT_FUNCTIONS
    #undef DECLARE_GL_FUNCTION
    #undef DECLARE_GL_FUNCTION_OPTIONAL
    #undef DECLARE_SYS_FUNCTION
    #undef DECLARE_SYS_GL_FUNCTION
    return true;
}

bool GLLibrary::initGraphicsLibrary()
{
    #define TEMPEST_EXTRACT_FUNCTIONS
    #define DECLARE_GL_FUNCTION(return_type, name, ...) GL_LOAD_FUNCTION(name)
    #define DECLARE_GL_FUNCTION_OPTIONAL(caps, return_type, name, ...) GL_LOAD_FUNCTION_OPTIONAL(caps, name)
    #define DECLARE_SYS_FUNCTION(return_type, name, ...)
    #define DECLARE_SYS_GL_FUNCTION(return_type, name, ...)
    #include "tempest/graphics/opengl-backend/gl-library.hh"
    #undef TEMPEST_EXTRACT_FUNCTIONS
    #undef DECLARE_GL_FUNCTION
    #undef DECLARE_GL_FUNCTION_OPTIONAL
    #undef DECLARE_SYS_FUNCTION
    #undef DECLARE_SYS_GL_FUNCTION
    
    return true;
}
}
