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
#include "tempest/utils/logging.hh"

namespace Tempest
{
OSWindowSystem::OSWindowSystem()
{
    WNDCLASSEX  wclass;

    wclass.cbSize = sizeof(WNDCLASSEX);
    wclass.style = CS_VREDRAW | CS_HREDRAW | CS_OWNDC;
    wclass.lpfnWndProc = nullptr;
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
        Log(LogLevel::Error, "Cannot register CarinaWindow class");
        return;
    }
}
}