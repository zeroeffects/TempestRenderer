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

#include "tempest/graphics/opengl-backend/gl-library.hh"

namespace Tempest
{
#ifdef _WIN32
#   define GL_LIB_NAME "opengl32.dll"
#else
#   define GL_LIB_NAME "libGL.so"
#endif

#ifdef _WIN32
PFNWGLMAKECURRENTPROC wglMakeCurrent = nullptr;
PFNWGLDELETECONTEXTPROC wglDeleteContext = nullptr;
PFNWGLCREATECONTEXTPROC wglCreateContext = nullptr;
PFNWGLGETPROCADDRESSPROC wglGetProcAddress = nullptr;

PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB = nullptr;
PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB = nullptr;
PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB = nullptr;
#elif defined(LINUX)
PFNGLXSWAPBUFFERSPROC glXSwapBuffers = nullptr;
PFNGLXDESTROYCONTEXTPROC glXDestroyContext = nullptr;
PFNGLXMAKECURRENTPROC glXMakeCurrent = nullptr;
PFNGLXGETPROCADDRESSPROC glXGetProcAddress = nullptr;
PFNGLXQUERYVERSIONPROC glXQueryVersion = nullptr;
PFNGLXCHOOSEFBCONFIGPROC glXChooseFBConfig = nullptr;
PFNGLXGETFBCONFIGATTRIBPROC glXGetFBConfigAttrib = nullptr;
PFNGLXGETVISUALFROMFBCONFIGPROC glXGetVisualFromFBConfig = nullptr;
PFNGLXQUERYEXTENSIONSSTRINGPROC glXQueryExtensionsString = nullptr;
PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = nullptr;
PFNGLXCREATENEWCONTEXTPROC glXCreateNewContext = nullptr;
#else
#   error "Unsupported platform"
#endif

PFNGLDRAWRANGEELEMENTSPROC glDrawRangeElements = nullptr;
PFNGLTEXIMAGE3DPROC glTexImage3D = nullptr;
PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D = nullptr;
PFNGLCOPYTEXSUBIMAGE3DPROC glCopyTexSubImage3D = nullptr;

PFNGLBLENDEQUATIONPROC glBlendEquation = nullptr;
PFNGLBLENDCOLORPROC glBlendColor = nullptr;

PFNGLACTIVETEXTUREPROC glActiveTexture = nullptr;
PFNGLCOMPRESSEDTEXIMAGE1DPROC glCompressedTexImage1D = nullptr;
PFNGLCOMPRESSEDTEXIMAGE2DPROC glCompressedTexImage2D = nullptr;
PFNGLCOMPRESSEDTEXIMAGE3DPROC glCompressedTexImage3D = nullptr;
PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC glCompressedTexSubImage1D = nullptr;
PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC glCompressedTexSubImage2D = nullptr;
PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC glCompressedTexSubImage3D = nullptr;
PFNGLGETCOMPRESSEDTEXIMAGEPROC glGetCompressedTexImage = nullptr;
PFNGLSAMPLECOVERAGEPROC glSampleCoverage = nullptr;
PFNGLVIEWPORTPROC glViewport = nullptr;
PFNGLCLEARCOLORPROC glClearColor = nullptr;
PFNGLCLEARPROC glClear = nullptr;
PFNGLCOLORMASKPROC glColorMask = nullptr;
PFNGLBLENDFUNCPROC glBlendFunc = nullptr;
PFNGLLOGICOPPROC glLogicOp = nullptr;
PFNGLCULLFACEPROC glCullFace = nullptr;
PFNGLFRONTFACEPROC glFrontFace = nullptr;
PFNGLLINEWIDTHPROC glLineWidth = nullptr;
PFNGLPOLYGONMODEPROC glPolygonMode = nullptr;
PFNGLPOLYGONOFFSETPROC glPolygonOffset = nullptr;
PFNGLSCISSORPROC glScissor = nullptr;
PFNGLDRAWBUFFERPROC glDrawBuffer = nullptr;
PFNGLREADBUFFERPROC glReadBuffer = nullptr;
PFNGLENABLEPROC glEnable = nullptr;
PFNGLDISABLEPROC glDisable = nullptr;
PFNGLISENABLEDPROC glIsEnabled = nullptr;
PFNGLGETBOOLEANVPROC glGetBooleanv = nullptr;
PFNGLGETDOUBLEVPROC glGetDoublev = nullptr;
PFNGLGETFLOATVPROC glGetFloatv = nullptr;
PFNGLGETINTEGERVPROC glGetIntegerv = nullptr;
PFNGLGETERRORPROC glGetError = nullptr;
PFNGLGETSTRINGPROC glGetString = nullptr;
PFNGLFINISHPROC glFinish = nullptr;
PFNGLFLUSHPROC glFlush = nullptr;
PFNGLHINTPROC glHint = nullptr;

PFNGLCLEARDEPTHPROC glClearDepth = nullptr;
PFNGLDEPTHFUNCPROC glDepthFunc = nullptr;
PFNGLDEPTHMASKPROC glDepthMask = nullptr;
PFNGLDEPTHRANGEPROC glDepthRange = nullptr;

PFNGLPIXELSTOREFPROC glPixelStoref = nullptr;
PFNGLPIXELSTOREIPROC glPixelStorei = nullptr;
PFNGLREADPIXELSPROC glReadPixels = nullptr;

PFNGLSTENCILFUNCPROC glStencilFunc = nullptr;
PFNGLSTENCILMASKPROC glStencilMask = nullptr;
PFNGLSTENCILOPPROC glStencilOp = nullptr;
PFNGLCLEARSTENCILPROC glClearStencil = nullptr;

PFNGLTEXPARAMETERFPROC glTexParameterf = nullptr;
PFNGLTEXPARAMETERIPROC glTexParameteri = nullptr;
PFNGLTEXPARAMETERFVPROC glTexParameterfv = nullptr;
PFNGLTEXPARAMETERIVPROC glTexParameteriv = nullptr;
PFNGLGETTEXPARAMETERFVPROC glGetTexParameterfv = nullptr;
PFNGLGETTEXPARAMETERIVPROC glGetTexParameteriv = nullptr;
PFNGLGETTEXLEVELPARAMETERFVPROC glGetTexLevelParameterfv = nullptr;
PFNGLGETTEXLEVELPARAMETERIVPROC glGetTexLevelParameteriv = nullptr;
PFNGLTEXIMAGE1DPROC glTexImage1D = nullptr;
PFNGLTEXIMAGE2DPROC glTexImage2D = nullptr;
PFNGLGETTEXIMAGEPROC glGetTexImage = nullptr;

PFNGLGENTEXTURESPROC glGenTextures = nullptr;
PFNGLDELETETEXTURESPROC glDeleteTextures = nullptr;
PFNGLBINDTEXTUREPROC glBindTexture = nullptr;
PFNGLISTEXTUREPROC glIsTexture = nullptr;

PFNGLTEXSUBIMAGE1DPROC glTexSubImage1D = nullptr;
PFNGLTEXSUBIMAGE2DPROC glTexSubImage2D = nullptr;
PFNGLCOPYTEXIMAGE1DPROC glCopyTexImage1D = nullptr;
PFNGLCOPYTEXIMAGE2DPROC glCopyTexImage2D = nullptr;
PFNGLCOPYTEXSUBIMAGE1DPROC glCopyTexSubImage1D = nullptr;
PFNGLCOPYTEXSUBIMAGE2DPROC glCopyTexSubImage2D = nullptr;

PFNGLBLENDFUNCSEPARATEPROC glBlendFuncSeparate = nullptr;
PFNGLMULTIDRAWARRAYSPROC glMultiDrawArrays = nullptr;
PFNGLMULTIDRAWELEMENTSPROC glMultiDrawElements = nullptr;
PFNGLMULTIDRAWELEMENTSINDIRECTPROC glMultiDrawElementsIndirect = nullptr;
PFNGLPOINTPARAMETERFPROC glPointParameterf = nullptr;
PFNGLPOINTPARAMETERFVPROC glPointParameterfv = nullptr;
PFNGLPOINTPARAMETERIPROC glPointParameteri = nullptr;
PFNGLPOINTPARAMETERIVPROC glPointParameteriv = nullptr;
PFNGLGENQUERIESPROC glGenQueries = nullptr;
PFNGLDELETEQUERIESPROC glDeleteQueries = nullptr;
PFNGLISQUERYPROC glIsQuery = nullptr;
PFNGLBEGINQUERYPROC glBeginQuery = nullptr;
PFNGLENDQUERYPROC glEndQuery = nullptr;
PFNGLGETQUERYIVPROC glGetQueryiv = nullptr;
PFNGLGETQUERYOBJECTIVPROC glGetQueryObjectiv = nullptr;
PFNGLGETQUERYOBJECTUIVPROC glGetQueryObjectuiv = nullptr;
PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;
PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
PFNGLISBUFFERPROC glIsBuffer = nullptr;
PFNGLBUFFERDATAPROC glBufferData = nullptr;
PFNGLBUFFERSUBDATAPROC glBufferSubData = nullptr;
PFNGLGETBUFFERSUBDATAPROC glGetBufferSubData = nullptr;
PFNGLMAPBUFFERPROC glMapBuffer = nullptr;
PFNGLUNMAPBUFFERPROC glUnmapBuffer = nullptr;
PFNGLGETBUFFERPARAMETERIVPROC glGetBufferParameteriv = nullptr;
PFNGLGETBUFFERPOINTERVPROC glGetBufferPointerv = nullptr;
PFNGLBLENDEQUATIONSEPARATEPROC glBlendEquationSeparate = nullptr;
PFNGLDRAWBUFFERSPROC glDrawBuffers = nullptr;
PFNGLSTENCILOPSEPARATEPROC glStencilOpSeparate = nullptr;
PFNGLSTENCILFUNCSEPARATEPROC glStencilFuncSeparate = nullptr;
PFNGLSTENCILMASKSEPARATEPROC glStencilMaskSeparate = nullptr;
PFNGLATTACHSHADERPROC glAttachShader = nullptr;
PFNGLPROGRAMPARAMETERIPROC glProgramParameteri = nullptr;
PFNGLBINDATTRIBLOCATIONPROC glBindAttribLocation = nullptr;
PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
PFNGLCREATESHADERPROC glCreateShader = nullptr;
PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
PFNGLDELETESHADERPROC glDeleteShader = nullptr;
PFNGLDETACHSHADERPROC glDetachShader = nullptr;
PFNGLGETACTIVEATTRIBPROC glGetActiveAttrib = nullptr;
PFNGLGETACTIVEUNIFORMPROC glGetActiveUniform = nullptr;
PFNGLGETATTACHEDSHADERSPROC glGetAttachedShaders = nullptr;
PFNGLGETATTRIBLOCATIONPROC glGetAttribLocation = nullptr;
PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
PFNGLGETSHADERSOURCEPROC glGetShaderSource = nullptr;
PFNGLISPROGRAMPROC glIsProgram = nullptr;
PFNGLISSHADERPROC glIsShader = nullptr;
PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
PFNGLVALIDATEPROGRAMPROC glValidateProgram = nullptr;
PFNGLCOLORMASKIPROC glColorMaski = nullptr;
PFNGLGETBOOLEANI_VPROC glGetBooleani_v = nullptr;
PFNGLGETINTEGERI_VPROC glGetIntegeri_v = nullptr;
PFNGLENABLEIPROC glEnablei = nullptr;
PFNGLDISABLEIPROC glDisablei = nullptr;
PFNGLISENABLEDIPROC glIsEnabledi = nullptr;
PFNGLBEGINTRANSFORMFEEDBACKPROC glBeginTransformFeedback = nullptr;
PFNGLENDTRANSFORMFEEDBACKPROC glEndTransformFeedback = nullptr;
PFNGLBINDBUFFERRANGEPROC glBindBufferRange = nullptr;
PFNGLBINDBUFFERBASEPROC glBindBufferBase = nullptr;
PFNGLTRANSFORMFEEDBACKVARYINGSPROC glTransformFeedbackVaryings = nullptr;
PFNGLGETTRANSFORMFEEDBACKVARYINGPROC glGetTransformFeedbackVarying = nullptr;
PFNGLCLAMPCOLORPROC glClampColor = nullptr;
PFNGLBEGINCONDITIONALRENDERPROC glBeginConditionalRender = nullptr;
PFNGLENDCONDITIONALRENDERPROC glEndConditionalRender = nullptr;
PFNGLGETUNIFORMUIVPROC glGetUniformuiv = nullptr;
PFNGLBINDFRAGDATALOCATIONPROC glBindFragDataLocation = nullptr;
PFNGLGETFRAGDATALOCATIONPROC glGetFragDataLocation = nullptr;
PFNGLTEXPARAMETERIIVPROC glTexParameterIiv = nullptr;
PFNGLTEXPARAMETERIUIVPROC glTexParameterIuiv = nullptr;
PFNGLGETTEXPARAMETERIIVPROC glGetTexParameterIiv = nullptr;
PFNGLGETTEXPARAMETERIUIVPROC glGetTexParameterIuiv = nullptr;
PFNGLCLEARBUFFERIVPROC glClearBufferiv = nullptr;
PFNGLCLEARBUFFERUIVPROC glClearBufferuiv = nullptr;
PFNGLCLEARBUFFERFVPROC glClearBufferfv = nullptr;
PFNGLCLEARBUFFERFIPROC glClearBufferfi = nullptr;
PFNGLGETSTRINGIPROC glGetStringi = nullptr;
PFNGLISRENDERBUFFERPROC glIsRenderbuffer = nullptr;
PFNGLBINDRENDERBUFFERPROC glBindRenderbuffer = nullptr;
PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers = nullptr;
PFNGLGENRENDERBUFFERSPROC glGenRenderbuffers = nullptr;
PFNGLRENDERBUFFERSTORAGEPROC glRenderbufferStorage = nullptr;
PFNGLGETRENDERBUFFERPARAMETERIVPROC glGetRenderbufferParameteriv = nullptr;
PFNGLISFRAMEBUFFERPROC glIsFramebuffer = nullptr;
PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer = nullptr;
PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers = nullptr;
PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers = nullptr;
PFNGLCHECKFRAMEBUFFERSTATUSPROC glCheckFramebufferStatus = nullptr;
PFNGLFRAMEBUFFERTEXTURE1DPROC glFramebufferTexture1D = nullptr;
PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D = nullptr;
PFNGLFRAMEBUFFERTEXTURE3DPROC glFramebufferTexture3D = nullptr;
PFNGLFRAMEBUFFERRENDERBUFFERPROC glFramebufferRenderbuffer = nullptr;
PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC glGetFramebufferAttachmentParameteriv = nullptr;
PFNGLGENERATEMIPMAPPROC glGenerateMipmap = nullptr;
PFNGLBLITFRAMEBUFFERPROC glBlitFramebuffer = nullptr;
PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC glRenderbufferStorageMultisample = nullptr;
PFNGLFRAMEBUFFERTEXTURELAYERPROC glFramebufferTextureLayer = nullptr;

PFNGLGENSAMPLERSPROC glGenSamplers = nullptr;
PFNGLDELETESAMPLERSPROC glDeleteSamplers = nullptr;
PFNGLISSAMPLERPROC glIsSampler = nullptr;
PFNGLBINDSAMPLERPROC glBindSampler = nullptr;
PFNGLSAMPLERPARAMETERIPROC glSamplerParameteri = nullptr;
PFNGLSAMPLERPARAMETERIVPROC glSamplerParameteriv = nullptr;
PFNGLSAMPLERPARAMETERFPROC glSamplerParameterf = nullptr;
PFNGLSAMPLERPARAMETERFVPROC glSamplerParameterfv = nullptr;
PFNGLGETSAMPLERPARAMETERIVPROC glGetSamplerParameteriv = nullptr;
PFNGLGETSAMPLERPARAMETERFVPROC glGetSamplerParameterfv = nullptr;

PFNGLTEXIMAGE2DMULTISAMPLEPROC glTexImage2DMultisample = nullptr;
PFNGLTEXIMAGE3DMULTISAMPLEPROC glTexImage3DMultisample = nullptr;
PFNGLGETMULTISAMPLEFVPROC glGetMultisamplefv = nullptr;
PFNGLSAMPLEMASKIPROC glSampleMaski = nullptr;

PFNGLFENCESYNCPROC glFenceSync = nullptr;
PFNGLISSYNCPROC glIsSync = nullptr;
PFNGLDELETESYNCPROC glDeleteSync = nullptr;
PFNGLCLIENTWAITSYNCPROC glClientWaitSync = nullptr;
PFNGLWAITSYNCPROC glWaitSync = nullptr;
PFNGLGETINTEGER64VPROC glGetInteger64v = nullptr;
PFNGLGETSYNCIVPROC glGetSynciv = nullptr;

PFNGLBUFFERSTORAGEPROC glBufferStorage = nullptr;
PFNGLCLEARTEXIMAGEPROC glClearTexImage = nullptr;
PFNGLCLEARTEXSUBIMAGEPROC glClearTexSubImage = nullptr;
PFNGLBINDBUFFERSBASEPROC glBindBuffersBase = nullptr;
PFNGLBINDBUFFERSRANGEPROC glBindBuffersRange = nullptr;

PFNGLMAPBUFFERRANGEPROC glMapBufferRange = nullptr;

PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSNVPROC glMultiDrawArraysIndirectBindlessNV = nullptr;
PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSNVPROC glMultiDrawElementsIndirectBindlessNV = nullptr;

PFNGLMAKEBUFFERRESIDENTNVPROC glMakeBufferResidentNV = nullptr;
PFNGLMAKEBUFFERNONRESIDENTNVPROC glMakeBufferNonResidentNV = nullptr;
PFNGLISBUFFERRESIDENTNVPROC glIsBufferResidentNV = nullptr;
PFNGLMAKENAMEDBUFFERRESIDENTNVPROC glMakeNamedBufferResidentNV = nullptr;
PFNGLMAKENAMEDBUFFERNONRESIDENTNVPROC glMakeNamedBufferNonResidentNV = nullptr;
PFNGLISNAMEDBUFFERRESIDENTNVPROC glIsNamedBufferResidentNV = nullptr;
PFNGLGETBUFFERPARAMETERUI64VNVPROC glGetBufferParameterui64vNV = nullptr;
PFNGLGETNAMEDBUFFERPARAMETERUI64VNVPROC glGetNamedBufferParameterui64vNV = nullptr;
PFNGLGETINTEGERUI64VNVPROC glGetIntegerui64vNV = nullptr;
PFNGLUNIFORMUI64NVPROC glUniformui64NV = nullptr;
PFNGLUNIFORMUI64VNVPROC glUniformui64vNV = nullptr;
PFNGLPROGRAMUNIFORMUI64NVPROC glProgramUniformui64NV = nullptr;
PFNGLPROGRAMUNIFORMUI64VNVPROC glProgramUniformui64vNV = nullptr;

PFNGLVERTEXATTRIBFORMATPROC glVertexAttribFormat = nullptr;

PFNGLGETUNIFORMINDICESPROC glGetUniformIndices = nullptr;
PFNGLGETACTIVEUNIFORMSIVPROC glGetActiveUniformsiv = nullptr;
PFNGLGETACTIVEUNIFORMNAMEPROC glGetActiveUniformName = nullptr;
PFNGLGETUNIFORMBLOCKINDEXPROC glGetUniformBlockIndex = nullptr;
PFNGLGETACTIVEUNIFORMBLOCKIVPROC glGetActiveUniformBlockiv = nullptr;
PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC glGetActiveUniformBlockName = nullptr;
PFNGLUNIFORMBLOCKBINDINGPROC glUniformBlockBinding = nullptr;

PFNGLDEBUGMESSAGECALLBACKPROC glDebugMessageCallback = nullptr;

PFNGLENABLECLIENTSTATEPROC glEnableClientState = nullptr;
PFNGLDISABLECLIENTSTATEPROC glDisableClientState = nullptr;

PFNGLBINDVERTEXBUFFERPROC glBindVertexBuffer = nullptr;

PFNGLVERTEXATTRIBBINDINGPROC glVertexAttribBinding = nullptr;
PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArrayARB = nullptr;

PFNGLGENPROGRAMPIPELINESPROC glGenProgramPipelines = nullptr;
PFNGLUSEPROGRAMSTAGESPROC glUseProgramStages = nullptr;
PFNGLBINDPROGRAMPIPELINEPROC glBindProgramPipeline = nullptr;
PFNGLUNIFORMSUBROUTINESUIVPROC glUniformSubroutinesuiv = nullptr;

PFNGLGETSUBROUTINEINDEXPROC glGetSubroutineIndex = nullptr;

PFNGLTEXTUREPARAMETERIEXTPROC glTextureParameteriEXT = nullptr;
PFNGLTEXTUREPARAMETERFEXTPROC glTextureParameterfEXT = nullptr;
PFNGLTEXTUREPARAMETERFVEXTPROC glTextureParameterfvEXT = nullptr;

PFNGLMAKETEXTUREHANDLERESIDENTARBPROC glMakeTextureHandleResidentARB = nullptr;
PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC glMakeTextureHandleNonResidentARB = nullptr;
PFNGLGETTEXTUREHANDLEARBPROC glGetTextureHandleARB = nullptr;

string ConvertGLErrorToString(GLenum err)
{
    switch(err)
    {
    case GL_NO_ERROR:
        return "no error"; break;
    case GL_INVALID_ENUM:
        return "invalid enumerated argument"; break;
    case GL_INVALID_VALUE:
        return "invalid value"; break;
    case GL_INVALID_OPERATION:
        return "invalid operation"; break;
    case GL_INVALID_FRAMEBUFFER_OPERATION:
        return "framebuffer object is incomplete";
    case GL_OUT_OF_MEMORY:
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

BOOL w32hackChoosePixelFormatARB(HDC hDC, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats)
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

bool GLLibrary::initGLX()
{
    if(m_GLLib.loaded())
        return true;

    if(!m_GLLib.load(GL_LIB_NAME))
        return false;

#ifdef _WIN32
    GL_LIB_LOAD_FUNCTION(wglMakeCurrent);
    GL_LIB_LOAD_FUNCTION(wglDeleteContext);
    GL_LIB_LOAD_FUNCTION(wglCreateContext);
    GL_LIB_LOAD_FUNCTION(wglGetProcAddress);
    return true;
#elif defined(LINUX)
    GL_LIB_LOAD_FUNCTION(glXGetProcAddress);

    GL_LOAD_FUNCTION(glXSwapBuffers);
    GL_LOAD_FUNCTION(glXDestroyContext);
    GL_LOAD_FUNCTION(glXMakeCurrent);
    GL_LOAD_FUNCTION(glXQueryVersion);
    GL_LOAD_FUNCTION(glXChooseFBConfig);
    GL_LOAD_FUNCTION(glXGetFBConfigAttrib);
    GL_LOAD_FUNCTION(glXGetVisualFromFBConfig);
    GL_LOAD_FUNCTION(glXQueryExtensionsString);
    GL_LOAD_FUNCTION(glXCreateNewContext);
    return LoadGLFunction(m_GLLib, "glXCreateContextAttribsARB", glXCreateContextAttribsARB);
#else
#   error "Unsupported platform"
#endif
}

bool GLLibrary::initGL()
{
    GL_LOAD_FUNCTION(glDrawRangeElements);
    GL_LOAD_FUNCTION(glTexImage3D);
    GL_LOAD_FUNCTION(glTexSubImage3D);
    GL_LOAD_FUNCTION(glCopyTexSubImage3D);

    GL_LOAD_FUNCTION(glBlendEquation);
    GL_LOAD_FUNCTION(glBlendColor);

    GL_LOAD_FUNCTION(glActiveTexture);
    GL_LOAD_FUNCTION(glCompressedTexImage1D);
    GL_LOAD_FUNCTION(glCompressedTexImage2D);
    GL_LOAD_FUNCTION(glCompressedTexImage3D);
    GL_LOAD_FUNCTION(glCompressedTexSubImage1D);
    GL_LOAD_FUNCTION(glCompressedTexSubImage2D);
    GL_LOAD_FUNCTION(glCompressedTexSubImage3D);
    GL_LOAD_FUNCTION(glGetCompressedTexImage);
    GL_LOAD_FUNCTION(glSampleCoverage);
    GL_LOAD_FUNCTION(glViewport);
    GL_LOAD_FUNCTION(glClearColor);
    GL_LOAD_FUNCTION(glClear);
    GL_LOAD_FUNCTION(glColorMask);
    GL_LOAD_FUNCTION(glBlendFunc);
    GL_LOAD_FUNCTION(glLogicOp);
    GL_LOAD_FUNCTION(glCullFace);
    GL_LOAD_FUNCTION(glFrontFace);
    GL_LOAD_FUNCTION(glLineWidth);
    GL_LOAD_FUNCTION(glPolygonMode);
    GL_LOAD_FUNCTION(glPolygonOffset);
    GL_LOAD_FUNCTION(glScissor);
    GL_LOAD_FUNCTION(glDrawBuffer);
    GL_LOAD_FUNCTION(glReadBuffer);
    GL_LOAD_FUNCTION(glEnable);
    GL_LOAD_FUNCTION(glDisable);
    GL_LOAD_FUNCTION(glIsEnabled);
    GL_LOAD_FUNCTION(glGetBooleanv);
    GL_LOAD_FUNCTION(glGetDoublev);
    GL_LOAD_FUNCTION(glGetFloatv);
    GL_LOAD_FUNCTION(glGetIntegerv);
    GL_LOAD_FUNCTION(glGetError);
    GL_LOAD_FUNCTION(glGetString);
    GL_LOAD_FUNCTION(glFinish);
    GL_LOAD_FUNCTION(glFlush);
    GL_LOAD_FUNCTION(glHint);

    GL_LOAD_FUNCTION(glClearDepth);
    GL_LOAD_FUNCTION(glDepthFunc);
    GL_LOAD_FUNCTION(glDepthMask);
    GL_LOAD_FUNCTION(glDepthRange);

    GL_LOAD_FUNCTION(glPixelStoref);
    GL_LOAD_FUNCTION(glPixelStorei);
    GL_LOAD_FUNCTION(glReadPixels);

    GL_LOAD_FUNCTION(glStencilFunc);
    GL_LOAD_FUNCTION(glStencilMask);
    GL_LOAD_FUNCTION(glStencilOp);
    GL_LOAD_FUNCTION(glClearStencil);

    GL_LOAD_FUNCTION(glTexParameterf);
    GL_LOAD_FUNCTION(glTexParameteri);
    GL_LOAD_FUNCTION(glTexParameterfv);
    GL_LOAD_FUNCTION(glTexParameteriv);
    GL_LOAD_FUNCTION(glGetTexParameterfv);
    GL_LOAD_FUNCTION(glGetTexParameteriv);
    GL_LOAD_FUNCTION(glGetTexLevelParameterfv);
    GL_LOAD_FUNCTION(glGetTexLevelParameteriv);
    GL_LOAD_FUNCTION(glTexImage1D);
    GL_LOAD_FUNCTION(glTexImage2D);
    GL_LOAD_FUNCTION(glGetTexImage);

    GL_LOAD_FUNCTION(glGenTextures);
    GL_LOAD_FUNCTION(glDeleteTextures);
    GL_LOAD_FUNCTION(glBindTexture);
    GL_LOAD_FUNCTION(glIsTexture);

    GL_LOAD_FUNCTION(glTexSubImage1D);
    GL_LOAD_FUNCTION(glTexSubImage2D);
    GL_LOAD_FUNCTION(glCopyTexImage1D);
    GL_LOAD_FUNCTION(glCopyTexImage2D);
    GL_LOAD_FUNCTION(glCopyTexSubImage1D);
    GL_LOAD_FUNCTION(glCopyTexSubImage2D);

    GL_LOAD_FUNCTION(glBlendFuncSeparate);
    GL_LOAD_FUNCTION(glMultiDrawArrays);
    GL_LOAD_FUNCTION(glMultiDrawElements);
    GL_LOAD_FUNCTION(glMultiDrawElementsIndirect);
    GL_LOAD_FUNCTION(glPointParameterf);
    GL_LOAD_FUNCTION(glPointParameterfv);
    GL_LOAD_FUNCTION(glPointParameteri);
    GL_LOAD_FUNCTION(glPointParameteriv);
    GL_LOAD_FUNCTION(glGenQueries);
    GL_LOAD_FUNCTION(glDeleteQueries);
    GL_LOAD_FUNCTION(glIsQuery);
    GL_LOAD_FUNCTION(glBeginQuery);
    GL_LOAD_FUNCTION(glEndQuery);
    GL_LOAD_FUNCTION(glGetQueryiv);
    GL_LOAD_FUNCTION(glGetQueryObjectiv);
    GL_LOAD_FUNCTION(glGetQueryObjectuiv);
    GL_LOAD_FUNCTION(glBindBuffer);
    GL_LOAD_FUNCTION(glDeleteBuffers);
    GL_LOAD_FUNCTION(glGenBuffers);
    GL_LOAD_FUNCTION(glIsBuffer);
    GL_LOAD_FUNCTION(glBufferData);
    GL_LOAD_FUNCTION(glBufferSubData);
    GL_LOAD_FUNCTION(glGetBufferSubData);
    GL_LOAD_FUNCTION(glMapBuffer);
    GL_LOAD_FUNCTION(glUnmapBuffer);
    GL_LOAD_FUNCTION(glGetBufferParameteriv);
    GL_LOAD_FUNCTION(glGetBufferPointerv);
    GL_LOAD_FUNCTION(glBlendEquationSeparate);
    GL_LOAD_FUNCTION(glDrawBuffers);
    GL_LOAD_FUNCTION(glStencilOpSeparate);
    GL_LOAD_FUNCTION(glStencilFuncSeparate);
    GL_LOAD_FUNCTION(glStencilMaskSeparate);
    GL_LOAD_FUNCTION(glAttachShader);
    GL_LOAD_FUNCTION(glGetProgramiv);
    GL_LOAD_FUNCTION(glBindAttribLocation);
    GL_LOAD_FUNCTION(glCompileShader);
    GL_LOAD_FUNCTION(glCreateProgram);
    GL_LOAD_FUNCTION(glCreateShader);
    GL_LOAD_FUNCTION(glDeleteProgram);
    GL_LOAD_FUNCTION(glDeleteShader);
    GL_LOAD_FUNCTION(glDetachShader);
    GL_LOAD_FUNCTION(glGetActiveAttrib);
    GL_LOAD_FUNCTION(glGetActiveUniform);
    GL_LOAD_FUNCTION(glGetAttachedShaders);
    GL_LOAD_FUNCTION(glGetAttribLocation);
    GL_LOAD_FUNCTION(glGetProgramiv);
    GL_LOAD_FUNCTION(glGetProgramInfoLog);
    GL_LOAD_FUNCTION(glGetShaderiv);
    GL_LOAD_FUNCTION(glGetShaderInfoLog);
    GL_LOAD_FUNCTION(glGetShaderSource);
    GL_LOAD_FUNCTION(glIsProgram);
    GL_LOAD_FUNCTION(glIsShader);
    GL_LOAD_FUNCTION(glLinkProgram);
    GL_LOAD_FUNCTION(glShaderSource);
    GL_LOAD_FUNCTION(glUseProgram);
    GL_LOAD_FUNCTION(glValidateProgram);
    GL_LOAD_FUNCTION(glColorMaski);
    GL_LOAD_FUNCTION(glGetBooleani_v);
    GL_LOAD_FUNCTION(glGetIntegeri_v);
    GL_LOAD_FUNCTION(glEnablei);
    GL_LOAD_FUNCTION(glDisablei);
    GL_LOAD_FUNCTION(glIsEnabledi);
    GL_LOAD_FUNCTION(glBeginTransformFeedback);
    GL_LOAD_FUNCTION(glEndTransformFeedback);
    GL_LOAD_FUNCTION(glBindBufferRange);
    GL_LOAD_FUNCTION(glBindBufferBase);
    GL_LOAD_FUNCTION(glTransformFeedbackVaryings);
    GL_LOAD_FUNCTION(glGetTransformFeedbackVarying);
    GL_LOAD_FUNCTION(glClampColor);
    GL_LOAD_FUNCTION(glBeginConditionalRender);
    GL_LOAD_FUNCTION(glEndConditionalRender);
    GL_LOAD_FUNCTION(glGetUniformuiv);
    GL_LOAD_FUNCTION(glBindFragDataLocation);
    GL_LOAD_FUNCTION(glGetFragDataLocation);
    GL_LOAD_FUNCTION(glTexParameterIiv);
    GL_LOAD_FUNCTION(glTexParameterIuiv);
    GL_LOAD_FUNCTION(glGetTexParameterIiv);
    GL_LOAD_FUNCTION(glGetTexParameterIuiv);
    GL_LOAD_FUNCTION(glClearBufferiv);
    GL_LOAD_FUNCTION(glClearBufferuiv);
    GL_LOAD_FUNCTION(glClearBufferfv);
    GL_LOAD_FUNCTION(glClearBufferfi);
    GL_LOAD_FUNCTION(glGetStringi);
    GL_LOAD_FUNCTION(glIsRenderbuffer);
    GL_LOAD_FUNCTION(glBindRenderbuffer);
    GL_LOAD_FUNCTION(glDeleteRenderbuffers);
    GL_LOAD_FUNCTION(glGenRenderbuffers);
    GL_LOAD_FUNCTION(glRenderbufferStorage);
    GL_LOAD_FUNCTION(glGetRenderbufferParameteriv);
    GL_LOAD_FUNCTION(glIsFramebuffer);
    GL_LOAD_FUNCTION(glBindFramebuffer);
    GL_LOAD_FUNCTION(glDeleteFramebuffers);
    GL_LOAD_FUNCTION(glGenFramebuffers);
    GL_LOAD_FUNCTION(glCheckFramebufferStatus);
    GL_LOAD_FUNCTION(glFramebufferTexture1D);
    GL_LOAD_FUNCTION(glFramebufferTexture2D);
    GL_LOAD_FUNCTION(glFramebufferTexture3D);
    GL_LOAD_FUNCTION(glFramebufferRenderbuffer);
    GL_LOAD_FUNCTION(glGetFramebufferAttachmentParameteriv);
    GL_LOAD_FUNCTION(glGenerateMipmap);
    GL_LOAD_FUNCTION(glBlitFramebuffer);
    GL_LOAD_FUNCTION(glRenderbufferStorageMultisample);
    GL_LOAD_FUNCTION(glFramebufferTextureLayer);

    GL_LOAD_FUNCTION(glGenSamplers);
    GL_LOAD_FUNCTION(glDeleteSamplers);
    GL_LOAD_FUNCTION(glIsSampler);
    GL_LOAD_FUNCTION(glBindSampler);
    GL_LOAD_FUNCTION(glSamplerParameteri);
    GL_LOAD_FUNCTION(glSamplerParameteriv);
    GL_LOAD_FUNCTION(glSamplerParameterf);
    GL_LOAD_FUNCTION(glSamplerParameterfv);
    GL_LOAD_FUNCTION(glGetSamplerParameteriv);
    GL_LOAD_FUNCTION(glGetSamplerParameterfv);
    GL_LOAD_FUNCTION(glTexImage2DMultisample);
    GL_LOAD_FUNCTION(glTexImage3DMultisample);
    GL_LOAD_FUNCTION(glGetMultisamplefv);
    GL_LOAD_FUNCTION(glSampleMaski);

    GL_LOAD_FUNCTION(glFenceSync);
    GL_LOAD_FUNCTION(glIsSync);
    GL_LOAD_FUNCTION(glDeleteSync);
    GL_LOAD_FUNCTION(glClientWaitSync);
    GL_LOAD_FUNCTION(glWaitSync);
    GL_LOAD_FUNCTION(glGetInteger64v);
    GL_LOAD_FUNCTION(glGetSynciv);

    GL_LOAD_FUNCTION(glBufferStorage);
    GL_LOAD_FUNCTION(glClearTexImage);
    GL_LOAD_FUNCTION(glClearTexSubImage);
    GL_LOAD_FUNCTION(glBindBuffersBase);
    GL_LOAD_FUNCTION(glBindBuffersRange);
    
    GL_LOAD_FUNCTION(glMapBufferRange);
    
    GL_LOAD_FUNCTION(glMultiDrawArraysIndirectBindlessNV);
    GL_LOAD_FUNCTION(glMultiDrawElementsIndirectBindlessNV);
    
    GL_LOAD_FUNCTION(glMakeBufferResidentNV);
    GL_LOAD_FUNCTION(glMakeBufferNonResidentNV);
    GL_LOAD_FUNCTION(glIsBufferResidentNV);
    GL_LOAD_FUNCTION(glMakeNamedBufferResidentNV);
    GL_LOAD_FUNCTION(glMakeNamedBufferNonResidentNV);
    GL_LOAD_FUNCTION(glIsNamedBufferResidentNV);
    GL_LOAD_FUNCTION(glGetBufferParameterui64vNV);
    GL_LOAD_FUNCTION(glGetNamedBufferParameterui64vNV);
    GL_LOAD_FUNCTION(glGetIntegerui64vNV);
    GL_LOAD_FUNCTION(glUniformui64NV);
    GL_LOAD_FUNCTION(glUniformui64vNV);
    GL_LOAD_FUNCTION(glProgramUniformui64NV);
    GL_LOAD_FUNCTION(glProgramUniformui64vNV);
    
    GL_LOAD_FUNCTION(glVertexAttribFormat);
    
    GL_LOAD_FUNCTION(glGetUniformIndices);
    GL_LOAD_FUNCTION(glGetActiveUniformsiv);
    GL_LOAD_FUNCTION(glGetActiveUniformName);
    GL_LOAD_FUNCTION(glGetUniformBlockIndex);
    GL_LOAD_FUNCTION(glGetActiveUniformBlockiv);
    GL_LOAD_FUNCTION(glGetActiveUniformBlockName);
    GL_LOAD_FUNCTION(glUniformBlockBinding);
    
    GL_LOAD_FUNCTION(glDebugMessageCallback);
    
    GL_LOAD_FUNCTION(glEnableClientState);
    GL_LOAD_FUNCTION(glDisableClientState);

    GL_LOAD_FUNCTION(glBindVertexBuffer);
    GL_LOAD_FUNCTION(glVertexAttribBinding);
    GL_LOAD_FUNCTION(glEnableVertexAttribArrayARB);

    GL_LOAD_FUNCTION(glProgramParameteri);
    GL_LOAD_FUNCTION(glUseProgramStages);
    GL_LOAD_FUNCTION(glBindProgramPipeline);
    GL_LOAD_FUNCTION(glGenProgramPipelines);
    GL_LOAD_FUNCTION(glUniformSubroutinesuiv);
    GL_LOAD_FUNCTION(glGetSubroutineIndex);
    
    GL_LOAD_FUNCTION(glTextureParameteriEXT);
    GL_LOAD_FUNCTION(glTextureParameterfEXT);
    GL_LOAD_FUNCTION(glTextureParameterfvEXT);

    GL_LOAD_FUNCTION(glMakeTextureHandleResidentARB);
    GL_LOAD_FUNCTION(glMakeTextureHandleNonResidentARB);
    GL_LOAD_FUNCTION(glGetTextureHandleARB);
    
    return true;
}
}
