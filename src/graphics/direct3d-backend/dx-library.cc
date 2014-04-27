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

#include "carina/dx-library.hh"
#include "carina/dx-window-impl.hh"

namespace Tempest
{
PFN_D3D11_CREATE_DEVITGE_AND_SWAP_CHAIN D3D11CreateDeviceAndSwapChain_MS = nullptr;
//pD3DCompile D3DCompile_MS = nullptr;

void CheckHResult(HRESULT hr) 
{
    if(SUCCEEDED(hr))
        return;

    auto msg = CREATE_SCOPED(LPSTR, LocalFree);
    msg = nullptr;
    DWORD res = FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM |
                              FORMAT_MESSAGE_ALLOCATE_BUFFER |
                              FORMAT_MESSAGE_IGNORE_INSERTS,  
                              nullptr,
                              hr,
                              MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                              (LPTSTR)&msg,
                              0,
                              nullptr);
    if(res && msg)
    {
        THROW_EXCEPTION("DirectX: " + string(msg));
    }
    else
    {
        std::stringstream ss;
        ss << "DirectX: Generic error: 0x" << std::hex << (unsigned)hr;
        THROW_EXCEPTION(ss.str());
    }
}

DXLibrary::DXLibrary()
    :   RenderingLibrary("DirectX")
{
}

bool DXLibrary::check()
{
    Library d3dlib, d3dxlib;
    return d3dlib.load("d3d11.dll");
}

bool DXLibrary::init()
{
    if(m_D3D11Lib.loaded())
        return false;
    m_D3D11Lib.load("d3d11.dll");
    m_D3DCompilerLib.load(D3DCOMPILER_DLL_A);
    D3D_LOAD_FUNCTION(m_D3D11Lib, PFN_D3D11_CREATE_DEVITGE_AND_SWAP_CHAIN, D3D11CreateDeviceAndSwapChain);
    return true;
}
}
