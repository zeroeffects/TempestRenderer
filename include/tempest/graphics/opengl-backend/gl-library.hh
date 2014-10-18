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

#ifndef GL_LIBRARY_HH
#define GL_LIBRARY_HH

#include "tempest/utils/library.hh"
#include "tempest/utils/macros.hh"

#ifdef _WIN32
    #include <windows.h>
#endif

#include <GL/gl.h>

#include "GL/glext.h"

#ifdef _WIN32
#include "GL/wglext.h"
#else
#include <X11/Xlib.h>
#include <GL/glx.h>
#endif

#include <sstream>
#include <iomanip>

#include "tempest/utils/assert.hh"
#include "tempest/utils/logging.hh"

namespace Tempest
{
class RenderingLibrary;
class WindowInfo;

class GLLibrary
{
#ifdef _WIN32
    HGLRC   m_RC = nullptr;
#elif !defined(LINUX)
#   error "Unsupported platform"
#endif
    Library m_GLLib;
public:
    explicit GLLibrary()=default;
     ~GLLibrary();

    bool initGLX();
    bool initGL();
};


//////////////////
// WGL typedefs //
//////////////////
#ifdef _WIN32
typedef BOOL (APIENTRYP PFNWGLMAKECURRENTPROC)(HDC hdc, HGLRC hglrc);
typedef BOOL (APIENTRYP PFNWGLDELETECONTEXTPROC)(HGLRC hglrc);
typedef HGLRC (APIENTRYP PFNWGLCREATECONTEXTPROC)(HDC hdc);
typedef PROC (APIENTRYP PFNWGLGETPROCADDRESSPROC)(LPCSTR lpszProc);

#endif

///////////////////
// WGL functions //
///////////////////
#ifdef _WIN32
extern PFNWGLGETPROCADDRESSPROC wglGetProcAddress;
extern PFNWGLMAKECURRENTPROC wglMakeCurrent;
extern PFNWGLDELETECONTEXTPROC wglDeleteContext;
extern PFNWGLCREATECONTEXTPROC wglCreateContext;

extern PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB;
extern PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB;
extern PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB;

HGLRC w32hackCreateContextAttribs(HDC hDC, HGLRC hShareContext, const int *attribList);
#endif

//////////////////
// GLX typedefs //
//////////////////
#ifndef _WIN32
typedef void (APIENTRYP PFNGLXSWAPBUFFERSPROC)(Display* dpy, GLXDrawable drawable);
typedef void (APIENTRYP PFNGLXDESTROYCONTEXTPROC)(Display* dpy, GLXContext ctx);
typedef Bool (APIENTRYP PFNGLXMAKECURRENTPROC)(Display* dpy, GLXDrawable drawable, GLXContext ctx);
typedef const char* (APIENTRYP PFNGLXQUERYEXTENSIONSSTRINGPROC)(Display* dpy, int screen);
typedef ProcType (APIENTRYP PFNGLXGETPROCADDRESSPROC)(const GLubyte* procName);
typedef GLXContext (APIENTRYP PFNGLXCREATECONTEXTATTRIBSARBPROC)(Display *dpy, GLXFBConfig config, GLXContext share_context, Bool direct, const int *attrib_list);
typedef Bool (APIENTRYP PFNGLXQUERYVERSIONPROC)(Display* dpy, int* major, int* minor);
typedef GLXFBConfig* (APIENTRYP PFNGLXCHOOSEFBCONFIGPROC)(Display* dpy, int screen, const int* attrib_list, int* nelements);
typedef int (APIENTRYP PFNGLXGETFBCONFIGATTRIBPROC)(Display* dpy,GLXFBConfig config, int attribute, int* value);
typedef XVisualInfo* (APIENTRYP PFNGLXGETVISUALFROMFBCONFIGPROC)(Display* dpy, GLXFBConfig config);
typedef GLXContext (APIENTRYP PFNGLXCREATENEWCONTEXTPROC)( Display *dpy, GLXFBConfig config, int renderType, GLXContext shareList, Bool direct );
#endif

///////////////////
// GLX functions //
///////////////////
#ifndef _WIN32
extern PFNGLXSWAPBUFFERSPROC glXSwapBuffers;
extern PFNGLXDESTROYCONTEXTPROC glXDestroyContext;
extern PFNGLXMAKECURRENTPROC glXMakeCurrent;
extern PFNGLXGETPROCADDRESSPROC glXGetProcAddress;
extern PFNGLXQUERYVERSIONPROC glXQueryVersion;
extern PFNGLXCHOOSEFBCONFIGPROC glXChooseFBConfig;
extern PFNGLXGETFBCONFIGATTRIBPROC glXGetFBConfigAttrib;
extern PFNGLXGETVISUALFROMFBCONFIGPROC glXGetVisualFromFBConfig;
extern PFNGLXQUERYEXTENSIONSSTRINGPROC glXQueryExtensionsString;
extern PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB;
extern PFNGLXCREATENEWCONTEXTPROC glXCreateNewContext;
#endif


/////////////////////
// OpenGL typedefs //
/////////////////////
typedef void (APIENTRYP PFNGLCLEARCOLORPROC)( GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha );
typedef void (APIENTRYP PFNGLCLEARPROC)( GLbitfield mask );
typedef void (APIENTRYP PFNGLCOLORMASKPROC)( GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha );
typedef void (APIENTRYP PFNGLBLENDFUNCPROC)( GLenum sfactor, GLenum dfactor );
typedef void (APIENTRYP PFNGLLOGICOPPROC)( GLenum opcode );
typedef void (APIENTRYP PFNGLCULLFACEPROC)( GLenum mode );
typedef void (APIENTRYP PFNGLFRONTFACEPROC)( GLenum mode );
typedef void (APIENTRYP PFNGLLINEWIDTHPROC)( GLfloat width );
typedef void (APIENTRYP PFNGLPOLYGONMODEPROC)( GLenum face, GLenum mode );
typedef void (APIENTRYP PFNGLPOLYGONOFFSETPROC)( GLfloat factor, GLfloat units );
typedef void (APIENTRYP PFNGLSCISSORPROC)( GLint x, GLint y, GLsizei width, GLsizei height);
typedef void (APIENTRYP PFNGLDRAWBUFFERPROC)( GLenum mode );
typedef void (APIENTRYP PFNGLREADBUFFERPROC)( GLenum mode );
typedef void (APIENTRYP PFNGLENABLEPROC)( GLenum cap );
typedef void (APIENTRYP PFNGLDISABLEPROC)( GLenum cap );
typedef GLboolean (APIENTRYP PFNGLISENABLEDPROC)( GLenum cap );
typedef void (APIENTRYP PFNGLGETBOOLEANVPROC)( GLenum pname, GLboolean *params );
typedef void (APIENTRYP PFNGLGETDOUBLEVPROC)( GLenum pname, GLdouble *params );
typedef void (APIENTRYP PFNGLGETFLOATVPROC)( GLenum pname, GLfloat *params );
typedef void (APIENTRYP PFNGLGETINTEGERVPROC)( GLenum pname, GLint *params );
typedef GLenum (APIENTRYP PFNGLGETERRORPROC)( void );
typedef const GLubyte* (APIENTRYP PFNGLGETSTRINGPROC)( GLenum name );
typedef void (APIENTRYP PFNGLFINISHPROC)( void );
typedef void (APIENTRYP PFNGLFLUSHPROC)( void );
typedef void (APIENTRYP PFNGLHINTPROC)( GLenum target, GLenum mode );

typedef void (APIENTRYP PFNGLCLEARDEPTHPROC)( GLclampd depth );
typedef void (APIENTRYP PFNGLDEPTHFUNCPROC)( GLenum func );
typedef void (APIENTRYP PFNGLDEPTHMASKPROC)( GLboolean flag );
typedef void (APIENTRYP PFNGLDEPTHRANGEPROC)( GLclampd near_val, GLclampd far_val );

typedef void (APIENTRYP PFNGLPIXELSTOREFPROC)( GLenum pname, GLfloat param );
typedef void (APIENTRYP PFNGLPIXELSTOREIPROC)( GLenum pname, GLint param );
typedef void (APIENTRYP PFNGLREADPIXELSPROC)( GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid *pixels );

typedef void (APIENTRYP PFNGLSTENCILFUNCPROC)( GLenum func, GLint ref, GLuint mask );
typedef void (APIENTRYP PFNGLSTENCILMASKPROC)( GLuint mask );
typedef void (APIENTRYP PFNGLSTENCILOPPROC)( GLenum fail, GLenum zfail, GLenum zpass );
typedef void (APIENTRYP PFNGLCLEARSTENCILPROC)( GLint s );

typedef void (APIENTRYP PFNGLTEXPARAMETERFPROC)( GLenum target, GLenum pname, GLfloat param );
typedef void (APIENTRYP PFNGLTEXPARAMETERIPROC)( GLenum target, GLenum pname, GLint param );
typedef void (APIENTRYP PFNGLTEXPARAMETERFVPROC)( GLenum target, GLenum pname, const GLfloat *params );
typedef void (APIENTRYP PFNGLTEXPARAMETERIVPROC)( GLenum target, GLenum pname, const GLint *params );
typedef void (APIENTRYP PFNGLGETTEXPARAMETERFVPROC)( GLenum target, GLenum pname, GLfloat *params);
typedef void (APIENTRYP PFNGLGETTEXPARAMETERIVPROC)( GLenum target, GLenum pname, GLint *params );
typedef void (APIENTRYP PFNGLGETTEXLEVELPARAMETERFVPROC)( GLenum target, GLint level, GLenum pname, GLfloat *params );
typedef void (APIENTRYP PFNGLGETTEXLEVELPARAMETERIVPROC)( GLenum target, GLint level, GLenum pname, GLint *params );
typedef void (APIENTRYP PFNGLTEXIMAGE1DPROC)( GLenum target, GLint level, GLint internalFormat, GLsizei width, GLint border, GLenum format, GLenum type, const GLvoid *pixels );
typedef void (APIENTRYP PFNGLTEXIMAGE2DPROC)( GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels );
typedef void (APIENTRYP PFNGLGETTEXIMAGEPROC)( GLenum target, GLint level, GLenum format, GLenum type, GLvoid *pixels );

typedef void (APIENTRYP PFNGLGENTEXTURESPROC)( GLsizei n, GLuint *textures );
typedef void (APIENTRYP PFNGLDELETETEXTURESPROC)( GLsizei n, const GLuint *textures);
typedef void (APIENTRYP PFNGLBINDTEXTUREPROC)( GLenum target, GLuint texture );
typedef GLboolean (APIENTRYP PFNGLISTEXTUREPROC)( GLuint texture );

typedef void (APIENTRYP PFNGLTEXSUBIMAGE1DPROC)( GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const GLvoid *pixels );
typedef void (APIENTRYP PFNGLTEXSUBIMAGE2DPROC)( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *pixels );
typedef void (APIENTRYP PFNGLCOPYTEXIMAGE1DPROC)( GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border );
typedef void (APIENTRYP PFNGLCOPYTEXIMAGE2DPROC)( GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border );
typedef void (APIENTRYP PFNGLCOPYTEXSUBIMAGE1DPROC)( GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width );
typedef void (APIENTRYP PFNGLCOPYTEXSUBIMAGE2DPROC)( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height );

typedef void (APIENTRYP PFNGLVIEWPORTPROC)( GLint x, GLint y, GLsizei width, GLsizei height );

typedef void (APIENTRYP PFNGLDRAWRANGEELEMENTSPROC)( GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid *indices );
typedef void (APIENTRYP PFNGLTEXIMAGE3DPROC)( GLenum target, GLint level, GLenum internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *pixels );
typedef void (APIENTRYP PFNGLTEXSUBIMAGE3DPROC)( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid *pixels);
typedef void (APIENTRYP PFNGLCOPYTEXSUBIMAGE3DPROC)( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height );

typedef void (APIENTRYP PFNGLBLENDEQUATIONPROC)( GLenum mode );
typedef void (APIENTRYP PFNGLBLENDCOLORPROC)( GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha );

typedef void (APIENTRYP PFNGLACTIVETEXTUREPROC)( GLenum texture );
typedef void (APIENTRYP PFNGLCOMPRESSEDTEXIMAGE1DPROC)( GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const GLvoid *data );
typedef void (APIENTRYP PFNGLCOMPRESSEDTEXIMAGE2DPROC)( GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const GLvoid *data );
typedef void (APIENTRYP PFNGLCOMPRESSEDTEXIMAGE3DPROC)( GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid *data );
typedef void (APIENTRYP PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC)( GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const GLvoid *data );
typedef void (APIENTRYP PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC)( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const GLvoid *data );
typedef void (APIENTRYP PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC)( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid *data );
typedef void (APIENTRYP PFNGLGETCOMPRESSEDTEXIMAGEPROC)( GLenum target, GLint lod, GLvoid *img );
typedef void (APIENTRYP PFNGLSAMPLECOVERAGEPROC)( GLclampf value, GLboolean invert );

typedef void (APIENTRYP PFNGLENABLECLIENTSTATEPROC)(GLenum cap);
typedef void (APIENTRYP PFNGLDISABLECLIENTSTATEPROC)(GLenum cap);


//////////////////////
// OpenGL functions //
//////////////////////
extern PFNGLDRAWRANGEELEMENTSPROC glDrawRangeElements;
extern PFNGLTEXIMAGE3DPROC glTexImage3D;
extern PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D;
extern PFNGLCOPYTEXSUBIMAGE3DPROC glCopyTexSubImage3D;

extern PFNGLBLENDEQUATIONPROC glBlendEquation;
extern PFNGLBLENDCOLORPROC glBlendColor;

extern PFNGLACTIVETEXTUREPROC glActiveTexture;
extern PFNGLCOMPRESSEDTEXIMAGE1DPROC glCompressedTexImage1D;
extern PFNGLCOMPRESSEDTEXIMAGE2DPROC glCompressedTexImage2D;
extern PFNGLCOMPRESSEDTEXIMAGE3DPROC glCompressedTexImage3D;
extern PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC glCompressedTexSubImage1D;
extern PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC glCompressedTexSubImage2D;
extern PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC glCompressedTexSubImage3D;
extern PFNGLGETCOMPRESSEDTEXIMAGEPROC glGetCompressedTexImage;
extern PFNGLSAMPLECOVERAGEPROC glSampleCoverage;
extern PFNGLVIEWPORTPROC glViewport;
extern PFNGLCLEARCOLORPROC glClearColor;
extern PFNGLCLEARPROC glClear;
extern PFNGLCOLORMASKPROC glColorMask;
extern PFNGLBLENDFUNCPROC glBlendFunc;
extern PFNGLLOGICOPPROC glLogicOp;
extern PFNGLCULLFACEPROC glCullFace;
extern PFNGLFRONTFACEPROC glFrontFace;
extern PFNGLLINEWIDTHPROC glLineWidth;
extern PFNGLPOLYGONMODEPROC glPolygonMode;
extern PFNGLPOLYGONOFFSETPROC glPolygonOffset;
extern PFNGLSCISSORPROC glScissor;
extern PFNGLDRAWBUFFERPROC glDrawBuffer;
extern PFNGLREADBUFFERPROC glReadBuffer;
extern PFNGLENABLEPROC glEnable;
extern PFNGLDISABLEPROC glDisable;
extern PFNGLISENABLEDPROC glIsEnabled;
extern PFNGLGETBOOLEANVPROC glGetBooleanv;
extern PFNGLGETDOUBLEVPROC glGetDoublev;
extern PFNGLGETFLOATVPROC glGetFloatv;
extern PFNGLGETINTEGERVPROC glGetIntegerv;
extern PFNGLGETERRORPROC glGetError;
extern PFNGLGETSTRINGPROC glGetString;
extern PFNGLFINISHPROC glFinish;
extern PFNGLFLUSHPROC glFlush;
extern PFNGLHINTPROC glHint;

extern PFNGLENABLECLIENTSTATEPROC glEnableClientState;
extern PFNGLDISABLECLIENTSTATEPROC glDisableClientState;

extern PFNGLCLEARDEPTHPROC glClearDepth;
extern PFNGLDEPTHFUNCPROC glDepthFunc;
extern PFNGLDEPTHMASKPROC glDepthMask;
extern PFNGLDEPTHRANGEPROC glDepthRange;

extern PFNGLPIXELSTOREFPROC glPixelStoref;
extern PFNGLPIXELSTOREIPROC glPixelStorei;
extern PFNGLREADPIXELSPROC glReadPixels;

extern PFNGLSTENCILFUNCPROC glStencilFunc;
extern PFNGLSTENCILMASKPROC glStencilMask;
extern PFNGLSTENCILOPPROC glStencilOp;
extern PFNGLCLEARSTENCILPROC glClearStencil;

extern PFNGLTEXPARAMETERFPROC glTexParameterf;
extern PFNGLTEXPARAMETERIPROC glTexParameteri;
extern PFNGLTEXPARAMETERFVPROC glTexParameterfv;
extern PFNGLTEXPARAMETERIVPROC glTexParameteriv;
extern PFNGLGETTEXPARAMETERFVPROC glGetTexParameterfv;
extern PFNGLGETTEXPARAMETERIVPROC glGetTexParameteriv;
extern PFNGLGETTEXLEVELPARAMETERFVPROC glGetTexLevelParameterfv;
extern PFNGLGETTEXLEVELPARAMETERIVPROC glGetTexLevelParameteriv;
extern PFNGLTEXIMAGE1DPROC glTexImage1D;
extern PFNGLTEXIMAGE2DPROC glTexImage2D;
extern PFNGLGETTEXIMAGEPROC glGetTexImage;

extern PFNGLGENTEXTURESPROC glGenTextures;
extern PFNGLDELETETEXTURESPROC glDeleteTextures;
extern PFNGLBINDTEXTUREPROC glBindTexture;
extern PFNGLISTEXTUREPROC glIsTexture;

extern PFNGLTEXSUBIMAGE1DPROC glTexSubImage1D;
extern PFNGLTEXSUBIMAGE2DPROC glTexSubImage2D;
extern PFNGLCOPYTEXIMAGE1DPROC glCopyTexImage1D;
extern PFNGLCOPYTEXIMAGE2DPROC glCopyTexImage2D;
extern PFNGLCOPYTEXSUBIMAGE1DPROC glCopyTexSubImage1D;
extern PFNGLCOPYTEXSUBIMAGE2DPROC glCopyTexSubImage2D;

extern PFNGLBLENDFUNCSEPARATEPROC glBlendFuncSeparate;
extern PFNGLMULTIDRAWARRAYSPROC glMultiDrawArrays;
extern PFNGLMULTIDRAWELEMENTSPROC glMultiDrawElements;
extern PFNGLMULTIDRAWELEMENTSINDIRECTPROC glMultiDrawElementsIndirect;
extern PFNGLPOINTPARAMETERFPROC glPointParameterf;
extern PFNGLPOINTPARAMETERFVPROC glPointParameterfv;
extern PFNGLPOINTPARAMETERIPROC glPointParameteri;
extern PFNGLPOINTPARAMETERIVPROC glPointParameteriv;
extern PFNGLGENQUERIESPROC glGenQueries;
extern PFNGLDELETEQUERIESPROC glDeleteQueries;
extern PFNGLISQUERYPROC glIsQuery;
extern PFNGLBEGINQUERYPROC glBeginQuery;
extern PFNGLENDQUERYPROC glEndQuery;
extern PFNGLGETQUERYIVPROC glGetQueryiv;
extern PFNGLGETQUERYOBJECTIVPROC glGetQueryObjectiv;
extern PFNGLGETQUERYOBJECTUIVPROC glGetQueryObjectuiv;
extern PFNGLBINDBUFFERPROC glBindBuffer;
extern PFNGLDELETEBUFFERSPROC glDeleteBuffers;
extern PFNGLGENBUFFERSPROC glGenBuffers;
extern PFNGLISBUFFERPROC glIsBuffer;
extern PFNGLBUFFERDATAPROC glBufferData;
extern PFNGLBUFFERSUBDATAPROC glBufferSubData;
extern PFNGLGETBUFFERSUBDATAPROC glGetBufferSubData;
extern PFNGLMAPBUFFERPROC glMapBuffer;
extern PFNGLUNMAPBUFFERPROC glUnmapBuffer;
extern PFNGLGETBUFFERPARAMETERIVPROC glGetBufferParameteriv;
extern PFNGLGETBUFFERPOINTERVPROC glGetBufferPointerv;
extern PFNGLBLENDEQUATIONSEPARATEPROC glBlendEquationSeparate;
extern PFNGLDRAWBUFFERSPROC glDrawBuffers;
extern PFNGLSTENCILOPSEPARATEPROC glStencilOpSeparate;
extern PFNGLSTENCILFUNCSEPARATEPROC glStencilFuncSeparate;
extern PFNGLSTENCILMASKSEPARATEPROC glStencilMaskSeparate;
extern PFNGLATTACHSHADERPROC glAttachShader;
extern PFNGLPROGRAMPARAMETERIPROC glProgramParameteri;
extern PFNGLBINDATTRIBLOCATIONPROC glBindAttribLocation;
extern PFNGLCOMPILESHADERPROC glCompileShader;
extern PFNGLCREATEPROGRAMPROC glCreateProgram;
extern PFNGLCREATESHADERPROC glCreateShader;
extern PFNGLDELETEPROGRAMPROC glDeleteProgram;
extern PFNGLDELETESHADERPROC glDeleteShader;
extern PFNGLDETACHSHADERPROC glDetachShader;
extern PFNGLISPROGRAMPROC glIsProgram;
extern PFNGLISSHADERPROC glIsShader;
extern PFNGLLINKPROGRAMPROC glLinkProgram;
extern PFNGLSHADERSOURCEPROC glShaderSource;
extern PFNGLUSEPROGRAMPROC glUseProgram;
extern PFNGLVALIDATEPROGRAMPROC glValidateProgram;
extern PFNGLCOLORMASKIPROC glColorMaski;
extern PFNGLGETBOOLEANI_VPROC glGetBooleani_v;
extern PFNGLGETINTEGERI_VPROC glGetIntegeri_v;
extern PFNGLENABLEIPROC glEnablei;
extern PFNGLDISABLEIPROC glDisablei;
extern PFNGLISENABLEDIPROC glIsEnabledi;
extern PFNGLBEGINTRANSFORMFEEDBACKPROC glBeginTransformFeedback;
extern PFNGLENDTRANSFORMFEEDBACKPROC glEndTransformFeedback;
extern PFNGLBINDBUFFERRANGEPROC glBindBufferRange;
extern PFNGLBINDBUFFERBASEPROC glBindBufferBase;
extern PFNGLTRANSFORMFEEDBACKVARYINGSPROC glTransformFeedbackVaryings;
extern PFNGLGETTRANSFORMFEEDBACKVARYINGPROC glGetTransformFeedbackVarying;
extern PFNGLCLAMPCOLORPROC glClampColor;
extern PFNGLBEGINCONDITIONALRENDERPROC glBeginConditionalRender;
extern PFNGLENDCONDITIONALRENDERPROC glEndConditionalRender;
extern PFNGLBINDFRAGDATALOCATIONPROC glBindFragDataLocation;
extern PFNGLGETFRAGDATALOCATIONPROC glGetFragDataLocation;
extern PFNGLTEXPARAMETERIIVPROC glTexParameterIiv;
extern PFNGLTEXPARAMETERIUIVPROC glTexParameterIuiv;
extern PFNGLGETTEXPARAMETERIIVPROC glGetTexParameterIiv;
extern PFNGLGETTEXPARAMETERIUIVPROC glGetTexParameterIuiv;
extern PFNGLCLEARBUFFERIVPROC glClearBufferiv;
extern PFNGLCLEARBUFFERUIVPROC glClearBufferuiv;
extern PFNGLCLEARBUFFERFVPROC glClearBufferfv;
extern PFNGLCLEARBUFFERFIPROC glClearBufferfi;
extern PFNGLGETSTRINGIPROC glGetStringi;
extern PFNGLISRENDERBUFFERPROC glIsRenderbuffer;
extern PFNGLBINDRENDERBUFFERPROC glBindRenderbuffer;
extern PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers;
extern PFNGLGENRENDERBUFFERSPROC glGenRenderbuffers;
extern PFNGLRENDERBUFFERSTORAGEPROC glRenderbufferStorage;
extern PFNGLGETRENDERBUFFERPARAMETERIVPROC glGetRenderbufferParameteriv;
extern PFNGLISFRAMEBUFFERPROC glIsFramebuffer;
extern PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer;
extern PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers;
extern PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
extern PFNGLCHECKFRAMEBUFFERSTATUSPROC glCheckFramebufferStatus;
extern PFNGLFRAMEBUFFERTEXTURE1DPROC glFramebufferTexture1D;
extern PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D;
extern PFNGLFRAMEBUFFERTEXTURE3DPROC glFramebufferTexture3D;
extern PFNGLFRAMEBUFFERRENDERBUFFERPROC glFramebufferRenderbuffer;
extern PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC glGetFramebufferAttachmentParameteriv;
extern PFNGLGENERATEMIPMAPPROC glGenerateMipmap;
extern PFNGLBLITFRAMEBUFFERPROC glBlitFramebuffer;
extern PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC glRenderbufferStorageMultisample;
extern PFNGLFRAMEBUFFERTEXTURELAYERPROC glFramebufferTextureLayer;

extern PFNGLGENSAMPLERSPROC glGenSamplers;
extern PFNGLDELETESAMPLERSPROC glDeleteSamplers;
extern PFNGLISSAMPLERPROC glIsSampler;
extern PFNGLBINDSAMPLERPROC glBindSampler;
extern PFNGLSAMPLERPARAMETERIPROC glSamplerParameteri;
extern PFNGLSAMPLERPARAMETERIVPROC glSamplerParameteriv;
extern PFNGLSAMPLERPARAMETERFPROC glSamplerParameterf;
extern PFNGLSAMPLERPARAMETERFVPROC glSamplerParameterfv;
extern PFNGLGETSAMPLERPARAMETERIVPROC glGetSamplerParameteriv;
extern PFNGLGETSAMPLERPARAMETERFVPROC glGetSamplerParameterfv;

extern PFNGLTEXIMAGE2DMULTISAMPLEPROC glTexImage2DMultisample;
extern PFNGLTEXIMAGE3DMULTISAMPLEPROC glTexImage3DMultisample;
extern PFNGLGETMULTISAMPLEFVPROC glGetMultisamplefv;
extern PFNGLSAMPLEMASKIPROC glSampleMaski;

extern PFNGLFENCESYNCPROC glFenceSync;
extern PFNGLISSYNCPROC glIsSync;
extern PFNGLDELETESYNCPROC glDeleteSync;
extern PFNGLCLIENTWAITSYNCPROC glClientWaitSync;
extern PFNGLWAITSYNCPROC glWaitSync;
extern PFNGLGETINTEGER64VPROC glGetInteger64v;
extern PFNGLGETSYNCIVPROC glGetSynciv;

extern PFNGLBUFFERSTORAGEPROC glBufferStorage;
extern PFNGLCLEARTEXIMAGEPROC glClearTexImage;
extern PFNGLCLEARTEXSUBIMAGEPROC glClearTexSubImage;
extern PFNGLBINDBUFFERSBASEPROC glBindBuffersBase;
extern PFNGLBINDBUFFERSRANGEPROC glBindBuffersRange;

extern PFNGLMAPBUFFERRANGEPROC glMapBufferRange;

extern PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSNVPROC glMultiDrawArraysIndirectBindlessNV;
extern PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSNVPROC glMultiDrawElementsIndirectBindlessNV;

extern PFNGLMAKEBUFFERRESIDENTNVPROC glMakeBufferResidentNV;
extern PFNGLMAKEBUFFERNONRESIDENTNVPROC glMakeBufferNonResidentNV;
extern PFNGLISBUFFERRESIDENTNVPROC glIsBufferResidentNV;
extern PFNGLMAKENAMEDBUFFERRESIDENTNVPROC glMakeNamedBufferResidentNV;
extern PFNGLMAKENAMEDBUFFERNONRESIDENTNVPROC glMakeNamedBufferNonResidentNV;
extern PFNGLISNAMEDBUFFERRESIDENTNVPROC glIsNamedBufferResidentNV;
extern PFNGLGETBUFFERPARAMETERUI64VNVPROC glGetBufferParameterui64vNV;
extern PFNGLGETNAMEDBUFFERPARAMETERUI64VNVPROC glGetNamedBufferParameterui64vNV;
extern PFNGLGETINTEGERUI64VNVPROC glGetIntegerui64vNV;
extern PFNGLUNIFORMUI64NVPROC glUniformui64NV;
extern PFNGLUNIFORMUI64VNVPROC glUniformui64vNV;
extern PFNGLPROGRAMUNIFORMUI64NVPROC glProgramUniformui64NV;
extern PFNGLPROGRAMUNIFORMUI64VNVPROC glProgramUniformui64vNV;

extern PFNGLVERTEXATTRIBFORMATPROC glVertexAttribFormat;
extern PFNGLGETSHADERIVPROC glGetShaderiv;
extern PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
extern PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
extern PFNGLGETPROGRAMIVPROC glGetProgramiv;

extern PFNGLGETUNIFORMINDICESPROC glGetUniformIndices;
extern PFNGLGETACTIVEUNIFORMSIVPROC glGetActiveUniformsiv;
extern PFNGLGETACTIVEUNIFORMNAMEPROC glGetActiveUniformName;
extern PFNGLGETUNIFORMBLOCKINDEXPROC glGetUniformBlockIndex;
extern PFNGLGETACTIVEUNIFORMBLOCKIVPROC glGetActiveUniformBlockiv;
extern PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC glGetActiveUniformBlockName;
extern PFNGLUNIFORMBLOCKBINDINGPROC glUniformBlockBinding;

extern PFNGLDEBUGMESSAGECALLBACKPROC glDebugMessageCallback;

extern PFNGLBINDVERTEXBUFFERPROC glBindVertexBuffer;

extern PFNGLVERTEXATTRIBBINDINGPROC glVertexAttribBinding;
extern PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArrayARB;

extern PFNGLGENPROGRAMPIPELINESPROC glGenProgramPipelines;
extern PFNGLUSEPROGRAMSTAGESPROC glUseProgramStages;
extern PFNGLBINDPROGRAMPIPELINEPROC glBindProgramPipeline;
extern PFNGLUNIFORMSUBROUTINESUIVPROC glUniformSubroutinesuiv;

extern PFNGLGETSUBROUTINEINDEXPROC glGetSubroutineIndex;

extern PFNGLTEXTUREPARAMETERIEXTPROC glTextureParameteriEXT;
extern PFNGLTEXTUREPARAMETERFEXTPROC glTextureParameterfEXT;
extern PFNGLTEXTUREPARAMETERFVEXTPROC glTextureParameterfvEXT;

extern PFNGLMAKETEXTUREHANDLERESIDENTARBPROC glMakeTextureHandleResidentARB;
extern PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC glMakeTextureHandleNonResidentARB;
extern PFNGLGETTEXTUREHANDLEARBPROC glGetTextureHandleARB;

#ifdef _WIN32
#   define GL_GET_PROC_ADDRESS(func) wglGetProcAddress(func)
#elif defined(LINUX)
#   define GL_GET_PROC_ADDRESS(func) glXGetProcAddress(reinterpret_cast<const GLubyte*>(func))
#else
#	error "Unsupported platform"
#endif

#define GL_LIB_LOAD_FUNCTION(func) if(!LoadGLLibFunction(m_GLLib, TO_STRING(func), func)) return false;

template<typename TFunc>
bool LoadGLLibFunction(Library& gllib, const char* name, TFunc& func)
{
    func = reinterpret_cast<TFunc>(gllib.getProcAddress(name));
    return func != nullptr;
}

#define GL_LOAD_FUNCTION(func) if(!LoadGLFunction(m_GLLib, TO_STRING(func), func)) return false;

template<class TFunc>
bool LoadGLFunction(Library& gllib, const char* name, TFunc& func)
{
    func = reinterpret_cast<TFunc>(GL_GET_PROC_ADDRESS(name));
    if(!func)
    {
        func = reinterpret_cast<TFunc>(gllib.getProcAddress(name)); \
        return func != nullptr;
    }
    return true;
}
}

#endif /* GL_LIBRARY_HH */
