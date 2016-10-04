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
#include "tempest/graphics/opengl-backend/gl-window.hh"
#include "tempest/utils/assert.hh"
#include "tempest/utils/logging.hh"

namespace Tempest
{
struct ColorBufferDescription
{
    int RedChannelSize;
    int GreenChannelSize;
    int BlueChannelSize;
    int AlphaChannelSize;
};

struct DepthStencilBufferDescription
{
    int DepthSize;
    int StencilSize;
};

ColorBufferDescription TranslateColorBufferDescription(DataFormat data_fmt)
{
    switch(data_fmt)
    {
    case DataFormat::R16UNorm: return { 16, 0, 0, 0 };
    case DataFormat::RG16UNorm: return { 16, 16, 0, 0 };
//  case DataFormat::RGB16UNorm: return { 16, 16, 16, 0 };
    case DataFormat::RGBA16UNorm: return { 16, 16, 16, 16 };
    case DataFormat::R8UNorm: return { 8, 0, 0, 0 };
    case DataFormat::RG8UNorm: return { 8, 8, 0, 0 };
//  case DataFormat::RGB8UNorm: return { 8, 8, 8, 0 };
    case DataFormat::RGBA8UNorm: return { 8, 8, 8, 8 };
    default: TGE_ASSERT(false, "Unsupported data format.");
    }
    return { 8, 8, 8, 8};
}

DepthStencilBufferDescription TranslateDepthStencilBufferDescription(DataFormat data_fmt)
{
    switch(data_fmt)
    {
    case DataFormat::D16: return { 16, 0 };
    case DataFormat::D24S8: return { 24, 8 };
    case DataFormat::D32: return { 32, 0 };
    default: TGE_ASSERT(false, "Unsupported data format.");
    }
    return { 24, 8 };
}

GLWindow::~GLWindow()
{
    XDestroyWindow(m_Display->nativeHandle(), m_Window);
    if(m_Display && m_XColormap)
        XFreeColormap(m_Display->nativeHandle(), m_XColormap);
}

bool GLWindow::init(OSWindowSystem& wnd_sys, OSWindow parent, const WindowDescription& wdesc)
{
    ColorBufferDescription color_fmt = TranslateColorBufferDescription(wdesc.ColorBufferFormat);
    DepthStencilBufferDescription depth_fmt = TranslateDepthStencilBufferDescription(wdesc.DepthBufferFormat);
    
    m_Display = &wnd_sys;
    
    int vi_attr_list[] =
    {
        GLX_X_RENDERABLE,  True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE,   GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
        GLX_RED_SIZE,      color_fmt.RedChannelSize,
        GLX_GREEN_SIZE,    color_fmt.GreenChannelSize,
        GLX_BLUE_SIZE ,    color_fmt.BlueChannelSize,
        GLX_ALPHA_SIZE,    color_fmt.AlphaChannelSize,
        GLX_DEPTH_SIZE,    depth_fmt.DepthSize,
        GLX_STENCIL_SIZE,  depth_fmt.StencilSize,
        GLX_DOUBLEBUFFER,  wdesc.Buffering != BufferingType::Single ? 1 : 0,
        GLX_SAMPLES,       (int)wdesc.Samples,
        None
    };
    
    auto display = m_Display->nativeHandle();

    if(!parent)
    {
        parent = RootWindow(display, DefaultScreen(display));
    }

    int glx_major, glx_minor;
    if(!glXQueryVersion(display, &glx_major, &glx_minor) || (glx_major == 1 && glx_minor < 3) || glx_major < 1)
    {
        Log(LogLevel::Error, "GLX version lower than 1.3 -- please check your graphics driver");
        return false;
    }
    
    int fbcount;
    GLXFBConfig* fbconf = glXChooseFBConfig(display, DefaultScreen(display), vi_attr_list, &fbcount);
    m_FBConfig = std::shared_ptr<GLXFBConfig>(fbconf, XFreeRAII());
    if(!m_FBConfig)
    {
        Log(LogLevel::Error, "the application has failed to retrieve a framebuffer config");
        return false;
    }

    auto vi = CREATE_SCOPED(XVisualInfo*, XFree);
    vi = glXGetVisualFromFBConfig(display, *m_FBConfig); 
    if(!vi)
    {
        Log(LogLevel::Error, "Cannot extract visual information from framebuffer config");
        return false;
    }

    XSetWindowAttributes swa;
    swa.event_mask = KeyPressMask | KeyReleaseMask | PointerMotionMask | StructureNotifyMask;
    swa.colormap = m_XColormap = XCreateColormap(display, parent, vi->visual, AllocNone);
    XInstallColormap(display, m_XColormap);

    m_WindowInformation.Width = wdesc.Width;
    m_WindowInformation.Height = wdesc.Height;
    m_Window = XCreateWindow(display, parent,
                             0, 0, wdesc.Width, wdesc.Height, 0, vi->depth, InputOutput,
                             vi->visual, CWColormap | CWEventMask, &swa);
    if(!m_Window)
    {
        Log(LogLevel::Error, "the application has failed to create a X11 window");
        return false;
    }
    
    XStoreName(display, m_Window, wdesc.Title.c_str());
}

void GLWindow::show()
{
    XMapWindow(m_Display->nativeHandle(), m_Window);
}

void GLWindow::captureMouse() {}
void GLWindow::releaseMouse() {}

void GLWindow::resize(uint32_t width, uint32_t height)
{
    XMoveResizeWindow(m_Display->nativeHandle(), m_Window, 0, 0, width, height);
}

void GLWindow::swapBuffers(int)
{
    auto display = m_Display->nativeHandle();
    glXSwapBuffers(display, m_Window);
    XFlush(display);
    XEvent ev;
    m_WindowInformation.MouseDeltaX = m_WindowInformation.MouseDeltaY = 0;
    while(XEventsQueued(display, QueuedAlready))
    {
        XNextEvent(display, &ev);
        switch(ev.type)
        {
        case ConfigureNotify:
        {
            if(m_Window == ev.xconfigure.window)
            {
                m_WindowInformation.Width = ev.xconfigure.width,
                m_WindowInformation.Height = ev.xconfigure.height;
            }
        } break;
        case MotionNotify:
        {
            auto prev_mouse_x = m_WindowInformation.MouseX;
            auto prev_mouse_y = m_WindowInformation.MouseY;

            m_WindowInformation.MouseX = (int32_t)ev.xmotion.x;
            m_WindowInformation.MouseY = m_WindowInformation.Height - (int32_t)ev.xmotion.y;
            m_WindowInformation.MouseDeltaX = m_WindowInformation.MouseX - prev_mouse_x;
            m_WindowInformation.MouseDeltaY = m_WindowInformation.MouseY - prev_mouse_y;
        } break;
        case DestroyNotify:
        {
            m_WindowInformation.Flags |= TEMPEST_WINDOW_STATE_DEAD;
        } break;
        default: break;
        }
    }
}
}