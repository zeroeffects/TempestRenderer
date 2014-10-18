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

#ifndef _GL_WINDOW_HH_
#define _GL_WINDOW_HH_

#include <memory>

#ifdef LINUX
    #include <GL/glx.h>
#endif

#include "tempest/utils/types.hh"
#include "tempest/graphics/os-window.hh"
#include "tempest/graphics/rendering-definitions.hh"

namespace Tempest
{
enum class BufferingType
{
    Single,
    Double,
    Triple
};
    
struct WindowDescription
{
    size_t          Width = 100,
                    Height = 100;
    string          Title = "Generic window";
    DataFormat      ColorBufferFormat = DataFormat::RGBA8UNorm;
    DataFormat      DepthBufferFormat = DataFormat::D24S8;
    size_t          Samples = 1;
    BufferingType   Buffering = BufferingType::Double;
};

#ifdef LINUX 
struct XFreeRAII
{
    void operator()(void* ptr) { XFree(ptr); }
};

typedef std::shared_ptr<GLXFBConfig> GLXFBConfigPtr;
#endif

class GLWindow
{
    OSWindow        m_Window    = OSWindow();    //!< Handle to the OS-specific window implementation.
#ifdef LINUX
    Colormap        m_XColormap = Colormap(); //!< Colormap used for gamma correction.
    GLXFBConfigPtr  m_FBConfig;
#endif
    OSWindowSystem* m_Display   = nullptr;
public:
    explicit GLWindow()=default;
     ~GLWindow();
    
    /*! \brief Initializes the window.
     * 
     *  This function creates a window object that is usable for rendering.
     */
    bool init(OSWindowSystem& wnd_sys, OSWindow parent, const WindowDescription& wdesc);

#ifdef LINUX
    GLXFBConfigPtr getFBConfig() { return m_FBConfig; }
#endif
    OSWindow getWindowId() {  return m_Window; }
    
    void show();
    
    void swapBuffers();
};

//! \brief The OpenGL rendering context. Makes it possible to attach the rendering system to different windows.
class GLContext
{
#ifdef LINUX
    // Because there is not a cannonical description what it should contain we pessimize the code.
    GLXFBConfigPtr  m_FBConfig;
    GLXContext      m_GLXContext = GLXContext();
#endif
    OSWindowSystem* m_Display    = nullptr;
public:
    explicit GLContext()=default;
     ~GLContext();
    
    /*! \brief Attaches the rendering context to the specified window.
     * 
     *  This function does the whole heavy-lifting when it comes to attaching a rendering context to particular
     *  window.
     * 
     *  \remarks Have in mind that the windows must have similar setting. Else it might not be possible to
     *           attach the rendering context.
     * 
     *  \param wnd_sys  the operating system specific window system.
     *  \param window   the window that is going to be attached to this rendering context.
     */
    bool attach(OSWindowSystem& wnd_sys, GLWindow& window);
};
}

#endif // _GL_WINDOW_HH_