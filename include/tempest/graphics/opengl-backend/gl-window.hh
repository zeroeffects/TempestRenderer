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
#else
    HDC               m_DC        = nullptr;
#endif
    OSWindowSystem*   m_Display   = nullptr;

    WindowInformation m_WindowInformation;
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
#elif defined(_WIN32)
    HDC getDC() { return m_DC;  }
#endif
    OSWindow getWindowId() {  return m_Window; }
    
    size_t getWidth() const { return m_WindowInformation.Width; }
    size_t getHeight() const { return m_WindowInformation.Height; }

    void show();
    
    void swapBuffers();
};
}

#endif // _GL_WINDOW_HH_