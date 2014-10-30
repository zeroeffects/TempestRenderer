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

#ifndef _OSWINDOW_SYSTEM_HH
#define _OSWINDOW_SYSTEM_HH

#include "tempest/graphics/rendering-definitions.hh"

#ifdef LINUX
#include <X11/Xlib.h>

namespace Tempest
{
class OSWindowSystem
{
    Display* m_Display;
public:
    explicit OSWindowSystem() { m_Display = XOpenDisplay(nullptr); }
     ~OSWindowSystem() { XCloseDisplay(m_Display); }
    
    OSWindowSystem(const OSWindowSystem&)=delete;
    OSWindowSystem& operator=(const OSWindowSystem&)=delete;
     
    operator bool() const { return m_Display != nullptr; }
    Display* nativeHandle() { return m_Display; }
};
    
typedef Window OSWindow;
}

#elif defined(_WIN32)
#include <windows.h>

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

class OSWindowSystem
{
public:
    explicit OSWindowSystem();
    ~OSWindowSystem()=default;

    OSWindowSystem(const OSWindowSystem&)=delete;
    OSWindowSystem& operator=(const OSWindowSystem&)=delete;

    operator bool() const { return true; }
    int nativeHandle() { return 0; }
};

typedef HWND OSWindow;
}

#else
#   error "Unsupported platform"
#endif

#endif // _OSWINDOW_SYSTEM_HH