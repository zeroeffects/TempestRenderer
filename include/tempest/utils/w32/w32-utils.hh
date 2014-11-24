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

#ifndef _TEMPEST_W32_UTILS_HH_
#define _TEMPEST_W32_UTILS_HH_

namespace Tempest
{
template<class T>
class ComRAII
{
    T* m_Ptr;
public:
    ComRAII() : m_Ptr(nullptr) {}
    //  ComRAII(T* p) : m_Ptr(p) {}
    ComRAII(const ComRAII& raii)
        : m_Ptr(raii.m_Ptr) {
        if (m_Ptr) m_Ptr->AddRef();
    }
    ~ComRAII()
    {
        release();
    }

    ComRAII& operator=(const ComRAII& raii)
    {
        release();
        m_Ptr = raii.m_Ptr;
        if (m_Ptr)
            m_Ptr->AddRef();
        return *this;
    }

    ComRAII& operator=(T* const ptr)
    {
        release();
        m_Ptr = ptr;
        return *this;
    }

    void release()
    {
        if (m_Ptr)
        {
            m_Ptr->Release();
            m_Ptr = nullptr;
        }
    }

    T* get() { return m_Ptr; }
    const T* get() const { return m_Ptr; }

    T** operator&(){ return &m_Ptr; }
    operator T*() const { return m_Ptr; }
    T& operator*() { return *m_Ptr; }
    T* operator->() { return m_Ptr; }

    T* const * operator&() const { return &m_Ptr; }
    const T& operator*() const { return *m_Ptr; }
    const T* operator->() const { return m_Ptr; }
};
}

#endif // _TEMPEST_W32_UTILS_HH_