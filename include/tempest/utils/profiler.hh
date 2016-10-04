/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2016 Zdravko Velinov
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

#ifndef _TEMPEST_PROFILER_HH_
#define _TEMPEST_PROFILER_HH_

#include "tempest/utils/timer.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/utils/config.hh"

#include <vector>
#include <thread>
#include <mutex>

#if !defined(__CUDACC__) || !defined(__device_builtin__)
#   define __device_builtin__
#endif

typedef __device_builtin__ struct CUevent_st *cudaEvent_t;

namespace Tempest
{
class Profiler;

class ProfilerEvent
{
    friend class Profiler;

    const char* m_Name;
    uint64_t    m_StartTime,
                m_EndTime;
public:
    ProfilerEvent(const char* name, uint64_t start_time)
        :   m_Name(name),
            m_StartTime(start_time) {}

    uint64_t elapsedTime() const { return m_EndTime - m_StartTime; }
};

struct ThreadData
{
    std::thread::native_handle_type  Handle;
    std::vector<ProfilerEvent>       Events;
};

struct EventId
{
    uint32_t                        ThreadIndex;
    uint32_t                        EventIndex;
};

struct CUDAEvent
{
    const char*             Name;
    cudaEvent_t             Start = nullptr,
                            End = nullptr;
    uint64_t                ElapsedTime = 0;
};

struct CUDAEventBuffer
{
    std::vector<CUDAEvent>  Events;
    size_t                  EventIndex = 0,
                            ProcessedIndex = 0;
};

class Profiler: public Singleton<Profiler>
{
    static const size_t BufferCount = 2;

    TimeQuery               m_Timer;
    std::vector<ThreadData> m_Threads;
    uint32_t                m_Status = 0;
    uint64_t                m_UpdateRate = 5000000;
    uint64_t                m_LastUpdate;
    uint64_t                m_StartFrame;
    CUDAEventBuffer         m_CudaEvents;
    cudaEvent_t             m_CudaStart = nullptr,
                            m_CudaEnd = nullptr;
    
    std::mutex              m_Mutex;

public:
    Profiler();
    ~Profiler();

    void beginFrame();
    void endFrame();

    static void setPrintMode(bool enable);

    EventId beginCPUEvent(const char* name);
    void endCPUEvent(EventId evt_id);

    size_t beginGPUEvent(const char* name);
    void endGPUEvent(size_t evt_id);

#ifndef DISABLE_CUDA
    size_t beginCUDAEvent(const char* name);
    void endCUDAEvent(size_t evt_id);
#endif
};

struct FrameScope
{
public:
    FrameScope() { auto prof = Profiler::getSingletonPtr(); if(prof) prof->beginFrame(); }
    ~FrameScope() { auto prof = Profiler::getSingletonPtr(); if(prof) prof->endFrame(); }
};

struct EventScope
{
    EventId  m_Event;
public:
    EventScope(const char* name) { auto prof = Profiler::getSingletonPtr(); if(prof) m_Event = prof->beginCPUEvent(name); }
    ~EventScope() { auto prof = Profiler::getSingletonPtr(); if(prof) prof->endCPUEvent(m_Event); }
};

#ifndef DISABLE_CUDA
struct CUDAEventScope
{
    size_t  m_Event;
public:
    CUDAEventScope(const char* name) { auto prof = Profiler::getSingletonPtr(); if(prof) m_Event = prof->beginCUDAEvent(name); }
    ~CUDAEventScope() { auto prof = Profiler::getSingletonPtr(); if(prof) prof->endCUDAEvent(m_Event); }
};
#endif

struct GPUEventScope
{
    size_t  m_Event;
public:
    GPUEventScope(const char* name) { auto prof = Profiler::getSingletonPtr(); if(prof) m_Event = prof->beginGPUEvent(name); }
    ~GPUEventScope() { auto prof = Profiler::getSingletonPtr(); if(prof) prof->endGPUEvent(m_Event); }
};

#ifdef ENABLE_PROFILER
#   define FRAME_SCOPE() Tempest::FrameScope CONCAT_MACRO(__frame, __COUNTER__)
#   define CPU_EVENT_SCOPE(name) Tempest::EventScope CONCAT_MACRO(__evt, __COUNTER__)(name)
#   define CUDA_EVENT_SCOPE(name) Tempest::CUDAEventScope CONCAT_MACRO(__evt, __COUNTER__)(name)
#   define GPU_EVENT_SCOPE(name) Tempest::GPUEventScope CONCAT_MACRO(__evt, __COUNTER__)(name)
#else
#   define FRAME_SCOPE()
#   define CPU_EVENT_SCOPE(name)
#   define CUDA_EVENT_SCOPE(name)
#   define GPU_EVENT_SCOPE(name)
#endif

#if defined(ENABLE_PROFILER) && !defined(DISABLE_CUDA)
#   define CUDA_EVENT_SCOPE(name) Tempest::CUDAEventScope CONCAT_MACRO(__evt, __COUNTER__)(name)
#   define GPU_EVENT_SCOPE(name) Tempest::GPUEventScope CONCAT_MACRO(__evt, __COUNTER__)(name)
#else
#   define CUDA_EVENT_SCOPE(name)
#   define GPU_EVENT_SCOPE(name)
#endif
}

#endif // _TEMPEST_PROFILER_HH_