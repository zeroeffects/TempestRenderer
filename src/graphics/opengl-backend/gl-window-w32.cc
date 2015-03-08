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

#include "tempest/graphics/opengl-backend/gl-window.hh"
#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/utils/logging.hh"

namespace Tempest
{
struct DepthStencilBufferDescription
{
    int DepthSize;
    int StencilSize;
};

int TranslateColorBits(DataFormat data_fmt)
{
    switch(data_fmt)
    {
    case DataFormat::RGBA8UNorm: return 32;
    default: TGE_ASSERT(false, "Unsupported data format.");
    }
    return 32;
}

DepthStencilBufferDescription TranslateDepthStencilBufferDescription(DataFormat data_fmt)
{
    switch(data_fmt)
    {
    case DataFormat::D16: return{ 16, 0 };
    case DataFormat::D24S8: return{ 24, 8 };
    case DataFormat::D32: return{ 32, 0 };
    default: TGE_ASSERT(false, "Unsupported data format.");
    }
    return{ 24, 8 };
}

GLWindow::~GLWindow()
{
    if(m_DC)
    {
        ReleaseDC(m_Window, m_DC);
    }
}

extern string GetLastErrorString();

bool GLWindow::init(OSWindowSystem& wnd_sys, OSWindow parent, const WindowDescription& wdesc)
{
    auto ds_desc = TranslateDepthStencilBufferDescription(wdesc.DepthBufferFormat);
    const int pi_attr_list[] =
    {
        WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
        WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
        WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB,
        WGL_DOUBLE_BUFFER_ARB, wdesc.Buffering != BufferingType::Single ? GL_TRUE : GL_FALSE,
        WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
        WGL_COLOR_BITS_ARB, TranslateColorBits(wdesc.ColorBufferFormat),
        WGL_DEPTH_BITS_ARB, ds_desc.DepthSize,
        WGL_STENCIL_BITS_ARB, ds_desc.StencilSize,
        WGL_SAMPLES_ARB, static_cast<int>(wdesc.Samples > 1 ? wdesc.Samples : 0),
        0, 0
    };

    PIXELFORMATDESCRIPTOR pfd = 
    {
        sizeof(PIXELFORMATDESCRIPTOR),
        1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA,
        24,
        0, 0, 0, 0, 0, 0,
        8,
        0,
        0,
        0, 0, 0, 0,
        24,
        8,
        0,
        PFD_MAIN_PLANE,
        0,
        0, 0, 0,
    };

    int pixel_format;
    UINT num_format;

    DWORD dwStyle;
    if(parent)
        dwStyle = WS_CHILD;
    else
        dwStyle = WS_OVERLAPPEDWINDOW | WS_SYSMENU;

    m_Window = CreateWindowEx(0, "TempestWindow", wdesc.Title.c_str(), dwStyle,
                              0, 0, static_cast<int>(wdesc.Width), static_cast<int>(wdesc.Height),
                              parent, 0, (HINSTANCE)GetModuleHandle(NULL), this);
    if(!m_Window)
    {
        Log(LogLevel::Error, "Failed to create window");
        return false;
    }

    m_DC = GetDC(m_Window);

    if(!w32hackChoosePixelFormat(m_DC, pi_attr_list, NULL, 1, &pixel_format, &num_format))
    {
        Log(LogLevel::Error, "Failed to get pixel format");
        return false;
    }
    if(!SetPixelFormat(m_DC, pixel_format, &pfd))
    {
        Log(LogLevel::Error, "Invalid pixel format: ", GetLastErrorString());
        return false;
    }
    
    SetWindowLongPtr(m_Window, GWLP_USERDATA, reinterpret_cast<LONG>(&m_WindowInformation));

    return true;
}

void GLWindow::show()
{
    ShowWindow(m_Window, SW_SHOW);
}

void GLWindow::swapBuffers()
{
    MSG msg;
    SwapBuffers(m_DC);
    while(PeekMessage(&msg, m_Window, 0, 0, PM_REMOVE))
        DispatchMessage(&msg);
}
}