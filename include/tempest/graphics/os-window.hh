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
#include <cstdint>

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

namespace Tempest
{
enum
{
    TEMPEST_WINDOW_STATE_DEAD = 1 << 0,
    TEMPEST_WINDOW_STATE_ACTIVE = 1 << 1,
};

enum WindowEventType: uint16_t
{
    MouseMoved,
    MouseButtonPressed,
    MouseButtonReleased,
    MouseButtonDoubleClick,
    KeyPressed,
    KeyReleased,
    MouseScroll,
    Focus,
    Resize
};

enum class KeyboardKey: uint16_t
{
    Unknown,     //!< Unknown or unspecified key.
    BackSpace,   //!< Backspace key.
    Tab,         //!< Tab key.
    Enter,       //!< Enter key.
    LCtrl,       //!< Left Control key
    RCtrl,       //!< Right Control key.
    LShift,      //!< Left Shift key.
    RShift,      //!< Right Shift key.
    LAlt,        //!< Left Alt key.
    RAlt,        //!< Right Alt key.
    LWin,        //!< Left Windows/Super key.
    RWin,        //!< Right Windows/Super key.
    Pause,       //!< Pause key.
    CapsLock,    //!< Caps Lock key.
    Escape,      //!< Escape key.
    Space,       //!< Space bar.
    PageUp,      //!< Page Up key.
    PageDown,    //!< Page Down key.
    End,         //!< End key.
    Home,        //!< Home key.
    Left,        //!< Left arrow key.
    Right,       //!< Right arrow key.
    Up,          //!< Up arrow key.
    Down,        //!< Down arrow key.
    PrintScreen, //!< Print Screen key.
    Insert,      //!< Insert key.
    Delete,      //!< Delete key.
    Key_0,       //!< '0' key.
    Key_1,       //!< '1' key.
    Key_2,       //!< '2' key.
    Key_3,       //!< '3' key.
    Key_4,       //!< '4' key.
    Key_5,       //!< '5' key.
    Key_6,       //!< '6' key.
    Key_7,       //!< '7' key.
    Key_8,       //!< '8' key.
    Key_9,       //!< '9' key.
    Key_A,       //!< 'A' key.
    Key_B,       //!< 'B' key.
    Key_C,       //!< 'C' key.
    Key_D,       //!< 'D' key.
    Key_E,       //!< 'E' key.
    Key_F,       //!< 'F' key.
    Key_G,       //!< 'G' key.
    Key_H,       //!< 'H' key.
    Key_I,       //!< 'I' key.
    Key_J,       //!< 'J' key.
    Key_K,       //!< 'K' key.
    Key_L,       //!< 'L' key.
    Key_M,       //!< 'M' key.
    Key_N,       //!< 'N' key.
    Key_O,       //!< 'O' key.
    Key_P,       //!< 'P' key.
    Key_Q,       //!< 'Q' key.
    Key_R,       //!< 'R' key.
    Key_S,       //!< 'S' key.
    Key_T,       //!< 'T' key.
    Key_U,       //!< 'U' key.
    Key_V,       //!< 'V' key.
    Key_W,       //!< 'W' key.
    Key_X,       //!< 'X' key.
    Key_Y,       //!< 'Y' key.
    Key_Z,       //!< 'Z' key.
    NUM0,        //!< Numpad 0 key.
    NUM1,        //!< Numpad 1 key.
    NUM2,        //!< Numpad 2 key.
    NUM3,        //!< Numpad 3 key.
    NUM4,        //!< Numpad 4 key.
    NUM5,        //!< Numpad 5 key.
    NUM6,        //!< Numpad 6 key.
    NUM7,        //!< Numpad 7 key.
    NUM8,        //!< Numpad 8 key.
    NUM9,        //!< Numpad 9 key.
    F1,          //!< Function 1 key.
    F2,          //!< Function 2 key.
    F3,          //!< Function 3 key.
    F4,          //!< Function 4 key.
    F5,          //!< Function 5 key.
    F6,          //!< Function 6 key.
    F7,          //!< Function 7 key.
    F8,          //!< Function 8 key.
    F9,          //!< Function 9 key.
    F10,         //!< Function 10 key.
    F11,         //!< Function 11 key.
    F12,         //!< Function 12 key.
    NumLock,     //!< Num Lock key.
    ScrollLock   //!< Scroll Lock key.
};

struct MouseMovedEvent
{
    int32_t MouseX,
            MouseY,
            MouseDeltaX,
            MouseDeltaY;
};

struct WindowResizeEvent
{
    uint32_t Width,
             Height;
};

enum class MouseButtonId: uint16_t
{
    LeftButton,
    MiddleButton,
    RightButton,
    ExtraButton1,
    ExtraButton2,
    Count,
    Unknown
};

struct WindowSystemEvent
{
    WindowEventType             Type;
    union
    {
        uint16_t                EventModifier;
        uint16_t                Enabled;
        MouseButtonId           MouseButton;
        KeyboardKey             Key;
    };
    union
    {
        MouseMovedEvent         MouseMoved;
        WindowResizeEvent       WindowResize;
    };
};

enum CollectEvent
{
    COLLECT_MOUSE_EVENTS = 1 << 0,
	COLLECT_WINDOW_EVENTS = 1 << 1,
    COLLECT_KEYBOARDS_EVENT = 1 << 2,
};

struct WindowInformation
{
    uint32_t                       Width = 0,
                                   Height = 0,
                                   MouseButtons = 0,
                                   Flags = 0;
    int32_t                        MouseX = 0,
                                   MouseY = 0,
                                   MouseDeltaX = 0,
                                   MouseDeltaY = 0;
    uint32_t                       EventMask = 0;
    uint32_t                       MouseButtonStates = 0;
    uint32_t                       WindowStatus = 0;
    std::vector<WindowSystemEvent> EventQueue;
};

    
enum class BufferingType
{
    Single = 1,
    Double = 2,
    Triple = 3
};

struct WindowDescription
{
    uint32_t        Width = 100,
                    Height = 100;
    std::string     Title = "Generic window";
    DataFormat      ColorBufferFormat = DataFormat::RGBA8UNorm;
    DataFormat      DepthBufferFormat = DataFormat::D24S8;
    uint32_t        Samples = 1;
    BufferingType   Buffering = BufferingType::Double;
};

class Texture;
Texture* CaptureScreenshot(bool cursor = false);
Texture* CaptureScreenshot(OSWindow wnd, bool cursor = false);
void GenerateEvent(const WindowSystemEvent& evt);
void GetMousePosition(int32_t* x, int32_t* y);
}

#endif // _OSWINDOW_SYSTEM_HH