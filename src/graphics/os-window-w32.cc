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

#include "tempest/graphics/os-window.hh"
#include "tempest/graphics/opengl-backend/gl-window.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/logging.hh"

#include <Windowsx.h>

namespace Tempest
{
LRESULT CALLBACK TempestWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);

OSWindowSystem::OSWindowSystem()
{
    WNDCLASSEX  wclass;

    wclass.cbSize = sizeof(WNDCLASSEX);
    wclass.style = CS_VREDRAW | CS_HREDRAW | CS_OWNDC;
    wclass.lpfnWndProc = TempestWindowProc;
    wclass.cbClsExtra = 0;
    wclass.cbWndExtra = 0;
    wclass.hInstance = (HINSTANCE)GetModuleHandle(nullptr);
    wclass.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wclass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    wclass.lpszMenuName = nullptr;
    wclass.lpszClassName = "TempestWindow";
    wclass.hIconSm = LoadIcon(nullptr, IDI_WINLOGO);

    if(!RegisterClassEx(&wclass))
    {
        Log(LogLevel::Error, "Cannot register TempestWindow class");
        return;
    }
}

static void PushMouseButtonPressedEvent(WindowEventType mouse_evt_type, MouseButtonId button, std::vector<WindowSystemEvent>* wevent_queue)
{
    WindowSystemEvent wevent;
    wevent.Type = mouse_evt_type;
    wevent.MouseButton = button;
    wevent_queue->push_back(wevent);
}

MouseButtonId TranslateExtraButton(WPARAM wParam)
{
    switch(GET_XBUTTON_WPARAM(wParam))
    {
    default: TGE_ASSERT(false, "Unknown extra button");
    case XBUTTON1: return MouseButtonId::ExtraButton1;
    case XBUTTON2: return MouseButtonId::ExtraButton2; 
    }
}

inline static void SetMouseButtonPressed(MouseButtonId button, uint32_t* mouse_states)
{
    *mouse_states |= (1 << (uint32_t)button);
}

inline static void ClearMouseButtonPressed(MouseButtonId button, uint32_t* mouse_states)
{
    *mouse_states &= ~(1 << (uint32_t)button);
}

inline static KeyboardKey TranslateKey(int key)
{
    switch(key)
    {
    case VK_BACK: return KeyboardKey::BackSpace;
    case VK_TAB: return KeyboardKey::Tab;
    case VK_RETURN: return KeyboardKey::Enter;
    case VK_CONTROL:
    case VK_LCONTROL: return KeyboardKey::LCtrl;
    case VK_RCONTROL: return KeyboardKey::RCtrl;
    case VK_SHIFT:
    case VK_LSHIFT: return KeyboardKey::LShift;
    case VK_RSHIFT: return KeyboardKey::RShift;
    case VK_MENU:
    case VK_LMENU: return KeyboardKey::LAlt;
    case VK_RMENU: return KeyboardKey::RAlt;
    case VK_PAUSE: return KeyboardKey::Pause;
    case VK_CAPITAL: return KeyboardKey::CapsLock;
    case VK_ESCAPE: return KeyboardKey::Escape;
    case VK_SPACE: return KeyboardKey::Space;
    case VK_PRIOR: return KeyboardKey::PageUp;
    case VK_NEXT: return KeyboardKey::PageDown;
    case VK_END: return KeyboardKey::End;
    case VK_HOME: return KeyboardKey::Home;
    case VK_LEFT: return KeyboardKey::Left;
    case VK_RIGHT: return KeyboardKey::Right;
    case VK_UP: return KeyboardKey::Up;
    case VK_DOWN: return KeyboardKey::Down;
    case VK_SNAPSHOT: return KeyboardKey::PrintScreen;
    case VK_INSERT: return KeyboardKey::Insert;
    case VK_DELETE: return KeyboardKey::Delete;
    case '0': return KeyboardKey::Key_0;
    case '1': return KeyboardKey::Key_1;
    case '2': return KeyboardKey::Key_2;
    case '3': return KeyboardKey::Key_3;
    case '4': return KeyboardKey::Key_4;
    case '5': return KeyboardKey::Key_5;
    case '6': return KeyboardKey::Key_6;
    case '7': return KeyboardKey::Key_7;
    case '8': return KeyboardKey::Key_8;
    case '9': return KeyboardKey::Key_9;
    case 'A': return KeyboardKey::Key_A;
    case 'B': return KeyboardKey::Key_B;
    case 'C': return KeyboardKey::Key_C;
    case 'D': return KeyboardKey::Key_D;
    case 'E': return KeyboardKey::Key_E;
    case 'F': return KeyboardKey::Key_F;
    case 'G': return KeyboardKey::Key_G;
    case 'H': return KeyboardKey::Key_H;
    case 'I': return KeyboardKey::Key_I;
    case 'J': return KeyboardKey::Key_J;
    case 'K': return KeyboardKey::Key_K;
    case 'L': return KeyboardKey::Key_L;
    case 'M': return KeyboardKey::Key_M;
    case 'N': return KeyboardKey::Key_N;
    case 'O': return KeyboardKey::Key_O;
    case 'P': return KeyboardKey::Key_P;
    case 'Q': return KeyboardKey::Key_Q;
    case 'R': return KeyboardKey::Key_R;
    case 'S': return KeyboardKey::Key_S;
    case 'T': return KeyboardKey::Key_T;
    case 'U': return KeyboardKey::Key_U;
    case 'V': return KeyboardKey::Key_V;
    case 'W': return KeyboardKey::Key_W;
    case 'X': return KeyboardKey::Key_X;
    case 'Y': return KeyboardKey::Key_Y;
    case 'Z': return KeyboardKey::Key_Z;
    case VK_LWIN: return KeyboardKey::LWin;
    case VK_RWIN: return KeyboardKey::RWin;
    case VK_NUMPAD0: return KeyboardKey::NUM0;
    case VK_NUMPAD1: return KeyboardKey::NUM1;
    case VK_NUMPAD2: return KeyboardKey::NUM2;
    case VK_NUMPAD3: return KeyboardKey::NUM3;
    case VK_NUMPAD4: return KeyboardKey::NUM4;
    case VK_NUMPAD5: return KeyboardKey::NUM5;
    case VK_NUMPAD6: return KeyboardKey::NUM6;
    case VK_NUMPAD7: return KeyboardKey::NUM7;
    case VK_NUMPAD8: return KeyboardKey::NUM8;
    case VK_NUMPAD9: return KeyboardKey::NUM9;
    case VK_F1: return KeyboardKey::F1;
    case VK_F2: return KeyboardKey::F2;
    case VK_F3: return KeyboardKey::F3;
    case VK_F4: return KeyboardKey::F4;
    case VK_F5: return KeyboardKey::F5;
    case VK_F6: return KeyboardKey::F6;
    case VK_F7: return KeyboardKey::F7;
    case VK_F8: return KeyboardKey::F8;
    case VK_F9: return KeyboardKey::F9;
    case VK_F10: return KeyboardKey::F10;
    case VK_F11: return KeyboardKey::F11;
    case VK_F12: return KeyboardKey::F12;
    case VK_NUMLOCK: return KeyboardKey::NumLock;
    case VK_SCROLL: return KeyboardKey::ScrollLock;
    }

    return static_cast<KeyboardKey>(-1);
}

inline static WORD TranslateKeyToVKEY(KeyboardKey key)
{
    switch(key)
    {
    case KeyboardKey::BackSpace: return VK_BACK;
    case KeyboardKey::Tab: return VK_TAB;
    case KeyboardKey::Enter: return VK_RETURN;
    case KeyboardKey::LCtrl: return VK_LCONTROL;
    case KeyboardKey::RCtrl: return VK_RCONTROL;
    case KeyboardKey::LShift: return VK_LSHIFT;
    case KeyboardKey::RShift: return VK_RSHIFT;
    case KeyboardKey::LAlt: return VK_LMENU;
    case KeyboardKey::RAlt: return VK_RMENU;
    case KeyboardKey::Pause: return VK_PAUSE;
    case KeyboardKey::CapsLock: return VK_CAPITAL;
    case KeyboardKey::Escape: return VK_ESCAPE;
    case KeyboardKey::Space: return VK_SPACE;
    case KeyboardKey::PageUp: return VK_PRIOR;
    case KeyboardKey::PageDown: return VK_NEXT;
    case KeyboardKey::End: return VK_END;
    case KeyboardKey::Home: return VK_HOME;
    case KeyboardKey::Left: return VK_LEFT;
    case KeyboardKey::Right: return VK_RIGHT;
    case KeyboardKey::Up: return VK_UP;
    case KeyboardKey::Down: return VK_DOWN;
    case KeyboardKey::PrintScreen: return VK_SNAPSHOT;
    case KeyboardKey::Insert: return VK_INSERT;
    case KeyboardKey::Delete: return VK_DELETE;
    case KeyboardKey::Key_0: return '0';
    case KeyboardKey::Key_1: return '1';
    case KeyboardKey::Key_2: return '2';
    case KeyboardKey::Key_3: return '3';
    case KeyboardKey::Key_4: return '4';
    case KeyboardKey::Key_5: return '5';
    case KeyboardKey::Key_6: return '6';
    case KeyboardKey::Key_7: return '7';
    case KeyboardKey::Key_8: return '8';
    case KeyboardKey::Key_9: return '9';
    case KeyboardKey::Key_A: return 'A';
    case KeyboardKey::Key_B: return 'B';
    case KeyboardKey::Key_C: return 'C';
    case KeyboardKey::Key_D: return 'D';
    case KeyboardKey::Key_E: return 'E';
    case KeyboardKey::Key_F: return 'F';
    case KeyboardKey::Key_G: return 'G';
    case KeyboardKey::Key_H: return 'H';
    case KeyboardKey::Key_I: return 'I';
    case KeyboardKey::Key_J: return 'J';
    case KeyboardKey::Key_K: return 'K';
    case KeyboardKey::Key_L: return 'L';
    case KeyboardKey::Key_M: return 'M';
    case KeyboardKey::Key_N: return 'N';
    case KeyboardKey::Key_O: return 'O';
    case KeyboardKey::Key_P: return 'P';
    case KeyboardKey::Key_Q: return 'Q';
    case KeyboardKey::Key_R: return 'R';
    case KeyboardKey::Key_S: return 'S';
    case KeyboardKey::Key_T: return 'T';
    case KeyboardKey::Key_U: return 'U';
    case KeyboardKey::Key_V: return 'V';
    case KeyboardKey::Key_W: return 'W';
    case KeyboardKey::Key_X: return 'X';
    case KeyboardKey::Key_Y: return 'Y';
    case KeyboardKey::Key_Z: return 'Z';
    case KeyboardKey::LWin: return VK_LWIN;
    case KeyboardKey::RWin: return VK_RWIN;
    case KeyboardKey::NUM0: return VK_NUMPAD0;
    case KeyboardKey::NUM1: return VK_NUMPAD1;
    case KeyboardKey::NUM2: return VK_NUMPAD2;
    case KeyboardKey::NUM3: return VK_NUMPAD3;
    case KeyboardKey::NUM4: return VK_NUMPAD4;
    case KeyboardKey::NUM5: return VK_NUMPAD5;
    case KeyboardKey::NUM6: return VK_NUMPAD6;
    case KeyboardKey::NUM7: return VK_NUMPAD7;
    case KeyboardKey::NUM8: return VK_NUMPAD8;
    case KeyboardKey::NUM9: return VK_NUMPAD9;
    case KeyboardKey::F1: return VK_F1;
    case KeyboardKey::F2: return VK_F2;
    case KeyboardKey::F3: return VK_F3;
    case KeyboardKey::F4: return VK_F4;
    case KeyboardKey::F5: return VK_F5;
    case KeyboardKey::F6: return VK_F6;
    case KeyboardKey::F7: return VK_F7;
    case KeyboardKey::F8: return VK_F8;
    case KeyboardKey::F9: return VK_F9;
    case KeyboardKey::F10: return VK_F10;
    case KeyboardKey::F11: return VK_F11;
    case KeyboardKey::F12: return VK_F12;
    case KeyboardKey::NumLock: return VK_NUMLOCK;
    case KeyboardKey::ScrollLock: return VK_SCROLL;
    }

    return -1;
}

LRESULT CALLBACK TempestWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    WindowInformation* info = reinterpret_cast<WindowInformation*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    if(info)
    {
        switch(message)
        {
        case WM_SIZE:
        {
            info->Width = LOWORD(lParam),
            info->Height = HIWORD(lParam);

            if(info->EventMask & COLLECT_WINDOW_EVENTS)
            {
                WindowSystemEvent evt;
                evt.Type = WindowEventType::Resize;
                evt.WindowResize.Width = info->Width;
                evt.WindowResize.Height = info->Height;

                info->EventQueue.push_back(evt);
            }

            return 0;
        } break;
        case WM_LBUTTONDOWN: PushMouseButtonPressedEvent(WindowEventType::MouseButtonPressed, MouseButtonId::LeftButton, &info->EventQueue); SetMouseButtonPressed(MouseButtonId::LeftButton, &info->MouseButtonStates); break;
        case WM_LBUTTONUP: PushMouseButtonPressedEvent(WindowEventType::MouseButtonReleased, MouseButtonId::LeftButton, &info->EventQueue); ClearMouseButtonPressed(MouseButtonId::LeftButton, &info->MouseButtonStates); break;
        case WM_LBUTTONDBLCLK: PushMouseButtonPressedEvent(WindowEventType::MouseButtonDoubleClick, MouseButtonId::LeftButton, &info->EventQueue); break;
        case WM_MBUTTONDOWN: PushMouseButtonPressedEvent(WindowEventType::MouseButtonPressed, MouseButtonId::MiddleButton, &info->EventQueue); SetMouseButtonPressed(MouseButtonId::MiddleButton, &info->MouseButtonStates); break;
        case WM_MBUTTONUP: PushMouseButtonPressedEvent(WindowEventType::MouseButtonReleased, MouseButtonId::MiddleButton, &info->EventQueue); ClearMouseButtonPressed(MouseButtonId::MiddleButton, &info->MouseButtonStates); break;
        case WM_MBUTTONDBLCLK: PushMouseButtonPressedEvent(WindowEventType::MouseButtonDoubleClick, MouseButtonId::MiddleButton, &info->EventQueue); break;
        case WM_RBUTTONDOWN: PushMouseButtonPressedEvent(WindowEventType::MouseButtonPressed, MouseButtonId::RightButton, &info->EventQueue); SetMouseButtonPressed(MouseButtonId::RightButton, &info->MouseButtonStates); break;
        case WM_RBUTTONUP: PushMouseButtonPressedEvent(WindowEventType::MouseButtonReleased, MouseButtonId::RightButton, &info->EventQueue); ClearMouseButtonPressed(MouseButtonId::RightButton, &info->MouseButtonStates); break;
        case WM_RBUTTONDBLCLK: PushMouseButtonPressedEvent(WindowEventType::MouseButtonDoubleClick, MouseButtonId::RightButton, &info->EventQueue); break;
        case WM_XBUTTONDOWN: PushMouseButtonPressedEvent(WindowEventType::MouseButtonPressed, TranslateExtraButton(wParam), &info->EventQueue); SetMouseButtonPressed(TranslateExtraButton(wParam), &info->MouseButtonStates); break;
        case WM_XBUTTONUP: PushMouseButtonPressedEvent(WindowEventType::MouseButtonReleased, TranslateExtraButton(wParam), &info->EventQueue); ClearMouseButtonPressed(TranslateExtraButton(wParam), &info->MouseButtonStates); break;
        case WM_XBUTTONDBLCLK: PushMouseButtonPressedEvent(WindowEventType::MouseButtonDoubleClick, TranslateExtraButton(wParam), &info->EventQueue); break;
        case WM_KEYDOWN:
        {
            WindowSystemEvent wevent;
            wevent.Type = WindowEventType::KeyPressed;
            wevent.Key = TranslateKey((int)wParam);
            info->EventQueue.push_back(wevent);
        } break;
        case WM_KEYUP:
        {
            WindowSystemEvent wevent;
            wevent.Type = WindowEventType::KeyReleased;
            wevent.Key = TranslateKey((int)wParam);
            info->EventQueue.push_back(wevent);
        } break;
        case WM_MOUSEMOVE:
        {
            auto prev_mouse_x = info->MouseX;
            auto prev_mouse_y = info->MouseY;

            info->MouseX = (int32_t)(short)GET_X_LPARAM(lParam);
            info->MouseY = (int32_t)(short)GET_Y_LPARAM(lParam);
            info->MouseDeltaX = info->MouseX - prev_mouse_x;
            info->MouseDeltaY = info->MouseY - prev_mouse_y;

            if(info->EventMask & COLLECT_MOUSE_EVENTS)
            {
                WindowSystemEvent wevent;
                wevent.Type = WindowEventType::MouseMoved;
                wevent.MouseMoved.MouseX = info->MouseX;
                wevent.MouseMoved.MouseY = info->MouseY;
                wevent.MouseMoved.MouseDeltaX = info->MouseDeltaX;
                wevent.MouseMoved.MouseDeltaY = info->MouseDeltaY;

                info->EventQueue.push_back(wevent);
            }
            return 0;
        } break;
		case WM_MOUSEWHEEL:
		{
            if(info->EventMask & COLLECT_MOUSE_EVENTS)
            {
                WindowSystemEvent wevent;
                wevent.Type = WindowEventType::MouseScroll;
                wevent.MouseMoved.MouseDeltaX = GET_WHEEL_DELTA_WPARAM(wParam);

                info->EventQueue.push_back(wevent);
            }
		} break;
        case WM_ACTIVATE:
        {
			if(info->EventMask & COLLECT_WINDOW_EVENTS)
            {
				WindowSystemEvent wevent;
                wevent.Type = WindowEventType::Focus;
				wevent.Enabled = (wParam != WA_INACTIVE);

                info->EventQueue.push_back(wevent);
			}

            if(wParam == WA_INACTIVE)
            {
                info->Flags &= ~TEMPEST_WINDOW_STATE_ACTIVE;
            }
            else
            {
                info->Flags |= TEMPEST_WINDOW_STATE_ACTIVE;
            }
        } break;
        case WM_QUIT:
        case WM_CLOSE:
        {
            info->Flags |= TEMPEST_WINDOW_STATE_DEAD;
            return 0;
        } break;
        default: break;
        }
    }

    return DefWindowProc(hwnd, message, wParam, lParam);
}

Texture* CaptureScreenshot(bool cursor)
{
    auto dc_screen = Tempest::CreateScoped<HDC>([](HDC dc) { ::ReleaseDC(nullptr, dc); });
    dc_screen = GetDC(nullptr);

    auto dc_mem = CREATE_SCOPED(HDC, ::DeleteObject);
    dc_mem = CreateCompatibleDC(dc_screen);

    if(!dc_mem)
    {
        return nullptr;
    }

    int left = GetSystemMetrics(SM_XVIRTUALSCREEN);
    int top = GetSystemMetrics(SM_YVIRTUALSCREEN);

    int height = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    int width = GetSystemMetrics(SM_CXVIRTUALSCREEN);

    auto bitmap = CREATE_SCOPED(HBITMAP, ::DeleteObject);
    bitmap = CreateCompatibleBitmap(dc_screen, width, height);
    
    if(!bitmap)
    {
        return nullptr;
    }

    SelectObject(dc_mem, bitmap);
    
    auto status = BitBlt(dc_mem, 0,0, width, height, 
                         dc_screen, left, top, SRCCOPY);
    
    if(!status)
    {
        return nullptr;
    }

    BITMAP bmp_screen;
    GetObject(bitmap, sizeof(BITMAP), &bmp_screen);

    BITMAPINFOHEADER   bi;
     
    bi.biSize = sizeof(BITMAPINFOHEADER);    
    bi.biWidth = bmp_screen.bmWidth;    
    bi.biHeight = bmp_screen.bmHeight;  
    bi.biPlanes = 1;    
    bi.biBitCount = 32;    
    bi.biCompression = BI_RGB;    
    bi.biSizeImage = 0;  
    bi.biXPelsPerMeter = 0;    
    bi.biYPelsPerMeter = 0;    
    bi.biClrUsed = 0;    
    bi.biClrImportant = 0;

    std::unique_ptr<uint32_t[]> tex_data(new uint32_t[bmp_screen.bmWidth * bmp_screen.bmHeight]);

    status = GetDIBits(dc_screen, bitmap, 0,
                       (UINT)bmp_screen.bmHeight,
                       tex_data.get(),
                       (BITMAPINFO *)&bi, DIB_RGB_COLORS);
    if(!status)
        return nullptr;

    for(size_t idx = 0, idx_end = bmp_screen.bmWidth * bmp_screen.bmHeight; idx < idx_end; ++idx)
    {
        uint32_t color = tex_data[idx];
        tex_data[idx] = rgba(rgbaB(color), rgbaG(color), rgbaR(color), rgbaA(color));
    }

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = static_cast<uint16_t>(bmp_screen.bmWidth);
    tex_desc.Height = static_cast<uint16_t>(bmp_screen.bmHeight);
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    return new Texture(tex_desc, reinterpret_cast<uint8_t*>(tex_data.release()));
}

Texture* CaptureScreenshot(OSWindow wnd, bool cursor)
{
    auto dc_screen = Tempest::CreateScoped<HDC>([](HDC dc) { ::ReleaseDC(nullptr, dc); });
    dc_screen = GetDC(nullptr);

    auto dc_mem = CREATE_SCOPED(HDC, ::DeleteObject);
    dc_mem = CreateCompatibleDC(dc_screen);

    if(!dc_mem)
    {
        return nullptr;
    }

    int left = GetSystemMetrics(SM_XVIRTUALSCREEN);
    int top = GetSystemMetrics(SM_YVIRTUALSCREEN);

    int height = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    int width = GetSystemMetrics(SM_CXVIRTUALSCREEN);

    RECT rect;
    GetWindowRect(wnd, &rect);

    rect.left = Clamp((int)rect.left, left, width);
    rect.right = Clamp((int)rect.right, left, width);
    rect.top = Clamp((int)rect.top, top, height);
    rect.bottom = Clamp((int)rect.bottom, top, height);

    size_t tex_width = rect.right - rect.left;
    size_t tex_height = rect.bottom - rect.top;

    auto bitmap = CREATE_SCOPED(HBITMAP, ::DeleteObject);
    bitmap = CreateCompatibleBitmap(dc_screen, tex_width, tex_height);
    
    if(!bitmap)
    {
        return nullptr;
    }

    auto old_bm = CREATE_SCOPED(HBITMAP, ::DeleteObject);
    SelectObject(dc_mem, bitmap);
    
    auto status = BitBlt(dc_mem, 0, 0, tex_width, tex_height, 
                         dc_screen, rect.left, rect.top, SRCCOPY);
    if(!status)
    {
        return nullptr;
    }

    if(cursor)
    {
        CURSORINFO pci;
        ICONINFO icon_info;

        pci.cbSize = sizeof(pci);
        GetCursorInfo(&pci);

        if(!GetIconInfo(pci.hCursor, &icon_info))
        {
            return nullptr;
        }

        auto color_cursor = CREATE_SCOPED(HBITMAP, ::DeleteObject);
        color_cursor = icon_info.hbmColor;
        auto mask_cursor = CREATE_SCOPED(HBITMAP, ::DeleteObject);
        mask_cursor = icon_info.hbmMask;

        auto cursor_dc = CREATE_SCOPED(HDC, ::DeleteObject);

        cursor_dc = CreateCompatibleDC(NULL);

        auto old_bm = CREATE_SCOPED(HBITMAP, ::DeleteObject);
        old_bm = (HBITMAP)SelectObject(cursor_dc, icon_info.hbmColor);

        BITMAP bm;
        GetObject(icon_info.hbmColor, sizeof(bm), &bm);

        POINT point;
        GetCursorPos(&point);

        MaskBlt(dc_mem, point.x - rect.left, point.y - rect.top, bm.bmWidth, bm.bmHeight,
                cursor_dc, 0, 0, icon_info.hbmMask, 0, 0, MAKEROP4(SRCPAINT,SRCCOPY));
    }

    BITMAP bmp_screen;
    GetObject(bitmap, sizeof(BITMAP), &bmp_screen);

    BITMAPINFOHEADER   bi;
     
    bi.biSize = sizeof(BITMAPINFOHEADER);    
    bi.biWidth = bmp_screen.bmWidth;    
    bi.biHeight = bmp_screen.bmHeight;  
    bi.biPlanes = 1;    
    bi.biBitCount = 32;    
    bi.biCompression = BI_RGB;    
    bi.biSizeImage = 0;  
    bi.biXPelsPerMeter = 0;    
    bi.biYPelsPerMeter = 0;    
    bi.biClrUsed = 0;    
    bi.biClrImportant = 0;

    std::unique_ptr<uint32_t[]> tex_data(new uint32_t[tex_width * tex_height]);

    status = GetDIBits(dc_screen, bitmap, 0,
                       (UINT)bmp_screen.bmHeight,
                       tex_data.get(),
                       (BITMAPINFO *)&bi, DIB_RGB_COLORS);
    if(!status)
        return nullptr;


    for(size_t y = 0; y < tex_height; ++y)
        for(size_t x = 0; x < tex_width; ++x)
        {
            uint32_t color = tex_data[y*bmp_screen.bmWidth + x];
            tex_data[y*tex_width + x] = rgba(rgbaB(color), rgbaG(color), rgbaR(color), rgbaA(color));
        }

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = static_cast<uint16_t>(tex_width);
    tex_desc.Height = static_cast<uint16_t>(tex_height);
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    return new Texture(tex_desc, reinterpret_cast<uint8_t*>(tex_data.release()));
}

void GenerateEvent(const WindowSystemEvent& evt)
{
    switch(evt.Type)
    {
    case WindowEventType::KeyPressed:
    {
        INPUT input;
        input.type = INPUT_KEYBOARD;
        input.ki.wVk = TranslateKeyToVKEY(evt.Key);
        input.ki.wScan = 0;
        input.ki.dwFlags = 0;
        input.ki.time = 0;
        input.ki.dwExtraInfo = GetMessageExtraInfo();

        SendInput(1, &input, sizeof(INPUT));
    } break;
    case WindowEventType::KeyReleased:
    {
        INPUT input;
        input.type = INPUT_KEYBOARD;
        input.ki.wVk = TranslateKeyToVKEY(evt.Key);
        input.ki.wScan = 0;
        input.ki.dwFlags = KEYEVENTF_KEYUP;
        input.ki.time = 0;
        input.ki.dwExtraInfo = GetMessageExtraInfo();

        SendInput(1, &input, sizeof(INPUT));
    } break;
    case WindowEventType::MouseButtonPressed:
    {
        INPUT input;
        input.type = INPUT_MOUSE;
        input.mi.dx = 0;
        input.mi.dy = 0;
        input.mi.mouseData = 0;
        switch(evt.MouseButton)
        {
        case MouseButtonId::LeftButton: input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN; break;
        case MouseButtonId::MiddleButton: input.mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN; break;
        case MouseButtonId::RightButton: input.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN; break;
        case MouseButtonId::ExtraButton1: input.mi.dwFlags = MOUSEEVENTF_XDOWN; input.mi.mouseData = XBUTTON1; break;
        case MouseButtonId::ExtraButton2: input.mi.dwFlags = MOUSEEVENTF_XDOWN; input.mi.mouseData = XBUTTON2; break;
        }
        
        input.mi.time = 0;
        input.mi.dwExtraInfo = GetMessageExtraInfo();

        SendInput(1, &input, sizeof(INPUT));
    } break;
    case WindowEventType::MouseButtonReleased:
    {
        INPUT input;
        input.type = INPUT_MOUSE;
        input.mi.dx = 0;
        input.mi.dy = 0;
        input.mi.mouseData = 0;
        switch(evt.MouseButton)
        {
        case MouseButtonId::LeftButton: input.mi.dwFlags = MOUSEEVENTF_LEFTUP; break;
        case MouseButtonId::MiddleButton: input.mi.dwFlags = MOUSEEVENTF_MIDDLEUP; break;
        case MouseButtonId::RightButton: input.mi.dwFlags = MOUSEEVENTF_RIGHTUP; break;
        case MouseButtonId::ExtraButton1: input.mi.dwFlags = MOUSEEVENTF_XUP; input.mi.mouseData = XBUTTON1; break;
        case MouseButtonId::ExtraButton2: input.mi.dwFlags = MOUSEEVENTF_XUP; input.mi.mouseData = XBUTTON2; break;
        }
        
        input.mi.time = 0;
        input.mi.dwExtraInfo = GetMessageExtraInfo();

        SendInput(1, &input, sizeof(INPUT));

    } break;
    case WindowEventType::MouseMoved:
    {
        INPUT input;
        input.type = INPUT_MOUSE;
        input.mi.dx = evt.MouseMoved.MouseX;
        input.mi.dy = evt.MouseMoved.MouseY;
        input.mi.mouseData = 0;
        input.mi.dwFlags = MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_MOVE|MOUSEEVENTF_MOVE_NOCOALESCE;
        input.mi.time = 0;
        input.mi.dwExtraInfo = GetMessageExtraInfo();

        SendInput(1, &input, sizeof(INPUT));
    } break;
    case WindowEventType::MouseScroll:
    {
        INPUT input;
        input.type = INPUT_MOUSE;
        input.mi.dx = 0;
        input.mi.dy = 0;
        input.mi.mouseData = evt.MouseMoved.MouseDeltaX;
        input.mi.dwFlags = MOUSEEVENTF_WHEEL;
        input.mi.time = 0;
        input.mi.dwExtraInfo = GetMessageExtraInfo();

        SendInput(1, &input, sizeof(INPUT));
    } break;
    }
}

void GetMousePosition(int32_t* x, int32_t* y)
{
    POINT point;
    GetCursorPos(&point);
    *x = ((point.x << 16) - 1)/GetSystemMetrics(SM_CXSCREEN);
    *y = ((point.y << 16) - 1)/GetSystemMetrics(SM_CYSCREEN);

    if(*x == 0 || *y == 0)
        DebugBreak();
}
}