/*   The MIT License
 *   
 *   Tempest Common
 *   Copyright (c) 2010-2011 Zdravko Velinov
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

#include "tempest/utils/memory.hh"
#include "tempest/utils/logging.hh"

namespace Tempest
{
#ifdef TGE_MEMORY_DEBUG
MemoryDebugger::MemoryDebugger()
{
    Log::message(TGE_LOG_DEBUG, "Memory debugging is enabled!");
}
MemoryDebugger::~MemoryDebugger()
{
    if(!m_Allocated.empty())
    {
        Log& log = Log::stream(TGE_LOG_DEBUG) << "The following memory addresses have not been deallocated:" << std::endl;
        for(PointerMap::iterator i = m_Allocated.begin(), iend = m_Allocated.end();
            i != iend; ++i)
            log << "Memory address: " << std::hex << i->first << "; allocated at: " << i->second << std::endl;
        Log::getSingleton().flush();
    }
}

bool MemoryDebugger::registerPointer(void* ptr, const string& info)
{
    for(PointerMap::iterator i = m_Allocated.begin(),
                             iend = m_Allocated.end(); i != iend; ++i)
    {
        if(i->first == ptr)
        {
            Log::stream(TGE_LOG_DEBUG) << "Unexpected behaviour. The following memory address is still in usage: " << std::hex << ptr << "\n"
                                      << Backtrace() << std::endl;
            Log::getSingleton().flush();
            return false;
        }
    }
    for(PointerMap::iterator j = m_Deallocated.begin(),
                             jend = m_Deallocated.end(); j != jend; ++j)
    {
        if(j->first == ptr)
        {
            Log::stream(TGE_LOG_DEBUG) << "Reusing memory address: " << std::hex << ptr << std::endl;
            m_Deallocated.erase(j);
            m_Allocated[ptr] = info;
            return true;
        }
    }
    
    m_Allocated[ptr] = info;
    return true;
}

bool MemoryDebugger::unregisterPointer(void* ptr, const string& info)
{
    for(PointerMap::iterator i = m_Allocated.begin(),
                             iend = m_Allocated.end(); i != iend; ++i)
    {
        if(i->first == ptr)
        {
            m_Deallocated[ptr] = info;
            m_Allocated.erase(i);
            return true;
        }
    }
    for(PointerMap::iterator j = m_Deallocated.begin(),
                             jend = m_Deallocated.end(); j != jend; ++j)
    {
        if(j->first == ptr)
        {
            Log::stream(TGE_LOG_DEBUG) << "The following memory address has already been deallocated: " << std::hex << ptr << "\n"
                                        << "\tTGE_DEALLOCATE has been used here: " << j->second << "\n"
                                        << Backtrace() << std::endl;
            Log::getSingleton().flush();
            return false;
        }
    }
    Log::stream(TGE_LOG_DEBUG) << "The following memory address has not been allocated by TGE_ALLOCATE: " << std::hex << ptr << "\n"
                              << Backtrace() << std::endl;
    Log::getSingleton().flush();
    return false;
}

bool MemoryDebugger::isAllocated(void* ptr)
{   
    for(PointerMap::iterator i = m_Allocated.begin(),
                             iend = m_Allocated.end(); i != iend; ++i)
        if(i->first == ptr)
            return true;
    return false;
}

string MemoryDebugger::getAllocatedInfo(void* ptr)
{
    for(PointerMap::iterator i = m_Allocated.begin(),
                             iend = m_Allocated.end(); i != iend; ++i)
        if(i->first == ptr)
            return i->second;
    return string();
}

string MemoryDebugger::getDeallocatedInfo(void* ptr)
{
    for(PointerMap::iterator i = m_Deallocated.begin(),
                             iend = m_Deallocated.end(); i != iend; ++i)
        if(i->first == ptr)
            return i->second;
    return string();
}
#endif
}