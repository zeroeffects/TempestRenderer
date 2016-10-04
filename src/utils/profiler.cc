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

#include "tempest/utils/profiler.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/system.hh"

#include <algorithm>
#include <iomanip>
#include <sstream>

#ifndef DISABLE_CUDA
#   include <cuda_runtime_api.h>
#endif

namespace Tempest
{
enum
{
    DISCARD_EVENT = 1 << 0,
    CAPTURE_PROFILE = 1 << 1,
    PRINT_REPORT = 1 << 2,
    REPORT_PRINTED = 1 << 3,
};

Profiler::Profiler()
{
#ifndef DISABLE_CUDA
    auto status = cudaEventCreate(&m_CudaStart);
    TGE_ASSERT(status == cudaSuccess, "Failed to create cuda event");

    status = cudaEventCreate(&m_CudaEnd);
    TGE_ASSERT(status == cudaSuccess, "Failed to create cuda event");
#endif
}

Profiler::~Profiler()
{
#ifndef DISABLE_CUDA
    for(auto& cuda_evt : m_CudaEvents.Events)
    {
        cudaEventDestroy(cuda_evt.Start);
        cudaEventDestroy(cuda_evt.End);
    }

    cudaEventDestroy(m_CudaStart);
    cudaEventDestroy(m_CudaEnd);
#endif
}

void Profiler::setPrintMode(bool enable)
{
    if(enable)
        Profiler::getSingleton().m_Status |= PRINT_REPORT|CAPTURE_PROFILE;
    else
        Profiler::getSingleton().m_Status &= ~PRINT_REPORT|CAPTURE_PROFILE;
}

void Profiler::beginFrame()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    if(m_Status & DISCARD_EVENT)
        return;

#ifndef DISABLE_CUDA
    auto status = cudaEventRecord(m_CudaStart, 0);
    TGE_ASSERT(status == cudaSuccess, "Failed to insert cuda event");
#endif   

    m_StartFrame = m_Timer.time();
}

void Profiler::endFrame()
{
    std::lock_guard<std::mutex> lock(m_Mutex);

    auto cur_time = m_Timer.time();
    if((m_Status & (DISCARD_EVENT|CAPTURE_PROFILE)) == (DISCARD_EVENT|CAPTURE_PROFILE))
    {
        bool gpu_data_ready = false, perform_capture = false;
        auto& evt_buffer = m_CudaEvents;
        auto event_count = evt_buffer.Events.size();

        if(m_LastUpdate - cur_time >= m_UpdateRate)
        {
            gpu_data_ready = true;
            perform_capture = true;
        }
        else
        {
        #ifndef DISABLE_CUDA
            for(; evt_buffer.ProcessedIndex < event_count; ++evt_buffer.ProcessedIndex)
            {
                auto& cuda_evt = evt_buffer.Events[evt_buffer.ProcessedIndex];
                auto status = cudaEventQuery(cuda_evt.End);
                if(status != cudaSuccess)
                    break;
            }

            if(evt_buffer.ProcessedIndex == event_count)
            {
                gpu_data_ready = true;
            }
        #endif
        }

        if((m_Status & (PRINT_REPORT|REPORT_PRINTED)) == PRINT_REPORT && gpu_data_ready)
        {
            std::stringstream ss;

            auto elapsed_time = (cur_time - m_StartFrame)*1e-3f;

            ss << "\nPROFILER REPORT:\n"
                  "================\n\n"
                  "Frame time: " << elapsed_time << "ms\n\n";

            for(auto& cur_thread : m_Threads)
            {
                auto thread_handle = cur_thread.Handle;
                auto& events = cur_thread.Events;
                auto thread_title = "Thread \"" + System::GetThreadName(thread_handle) + "\":\n";
                ss << thread_title
                   << std::setfill('=') << std::setw(thread_title.size()) << "\n";
                for(size_t idx = 0, idx_end = events.size(); idx < idx_end; ++idx)
                {
                    auto& evt = events[idx];
                    ss << idx + 1 << ". " << evt.m_Name << ": " << evt.elapsedTime()*1e-3f << "ms\n";
                }

                ss << "\n";

                events.clear();
            }

        #ifndef DISABLE_CUDA
            if(!m_CudaEvents.Events.empty())
            {
                float cuda_frame_time = 0.0f;
                auto status = cudaEventSynchronize(m_CudaEnd);
                TGE_ASSERT(status == cudaSuccess, "Failed to synchronize");

                status = cudaEventSynchronize(m_CudaStart);
                TGE_ASSERT(status == cudaSuccess, "Failed to synchronize");

                cudaEventElapsedTime(&cuda_frame_time, m_CudaStart, m_CudaEnd);

                static const char cuda_title[] = "CUDA Events:";
                ss << cuda_title << "\n"
                   << std::setfill('=') << std::setw(TGE_FIXED_ARRAY_SIZE(cuda_title)) << "\n\n"
                   << "CUDA frame time: " << cuda_frame_time << "ms\n";
                   
                for(size_t idx = 0, idx_end = m_CudaEvents.Events.size(); idx < idx_end; ++idx)
                {
                    auto& evt = m_CudaEvents.Events[idx];
                    float ms;

                    auto status = cudaEventSynchronize(evt.End);
                    TGE_ASSERT(status == cudaSuccess, "Failed to perform synchronization");

                    status = cudaEventElapsedTime(&ms, evt.Start, evt.End);
                    if(status != cudaSuccess)
                        continue;

                    if(ms > cuda_frame_time)
                    {
                        ms = 0.0f; // Garbage, it performs weirdly when a given section is empty
                    }

                    ss << idx + 1 << ". " << evt.Name << ": " << ms << "ms\n";
                }
            }
        #endif

            Tempest::Log(Tempest::LogLevel::Info, ss.str()); 
            m_Status |= REPORT_PRINTED;
        }

        if(perform_capture)
        {
            evt_buffer.ProcessedIndex = 0;
            evt_buffer.EventIndex = 0;

            for(auto& thread_data : m_Threads)
            {
                thread_data.Events.clear();
            }

            m_Status &= ~(DISCARD_EVENT|REPORT_PRINTED);
        }
    }
    else if(m_Status & CAPTURE_PROFILE)
    {
        m_LastUpdate = cur_time;

    #ifndef DISABLE_CUDA
        auto status = cudaEventRecord(m_CudaEnd, 0);
        TGE_ASSERT(status == cudaSuccess, "Failed to insert cuda event");
    #endif
        m_Status |= DISCARD_EVENT;
    }
}

size_t Profiler::beginGPUEvent(const char* name)
{
    return {};
}

void Profiler::endGPUEvent(size_t evt_id)
{

}

#ifndef DISABLE_CUDA
size_t Profiler::beginCUDAEvent(const char* name)
{
    if(m_Status & DISCARD_EVENT)
        return {};

    auto& evt_buffer = m_CudaEvents;

    auto evt_id = evt_buffer.EventIndex++;

    cudaEvent_t cuda_start_evt;
    if(evt_id == evt_buffer.Events.size())
    {
        CUDAEvent cuda_evt{};

        auto err = cudaEventCreate(&cuda_evt.Start);
        TGE_ASSERT(err == cudaSuccess, "Failed to create event");
        
        err = cudaEventCreate(&cuda_evt.End);
        TGE_ASSERT(err == cudaSuccess, "Failed to create event");

        cuda_evt.Name = name;
        cuda_start_evt = cuda_evt.Start;

        evt_buffer.Events.push_back(cuda_evt);
    }
    else
    {
        auto& cuda_evt = evt_buffer.Events.back();
        cuda_evt.Name = name;
        cuda_start_evt = cuda_evt.Start;
    }

    cudaEventRecord(cuda_start_evt, 0);

    return evt_id;
}

void Profiler::endCUDAEvent(size_t evt_id)
{
    if(m_Status & DISCARD_EVENT)
        return;

    auto& evt_buffer = m_CudaEvents;
    auto& cuda_evt = evt_buffer.Events[evt_id];
   
    cudaEventRecord(cuda_evt.End, 0);
}
#endif

EventId Profiler::beginCPUEvent(const char* name)
{
    std::lock_guard<std::mutex> lock(m_Mutex);

    if(m_Status & DISCARD_EVENT)
        return {};

    auto thread_handle = System::GetCurrentThreadNativeHandle();

    auto threads_begin = m_Threads.begin(),
         threads_end = m_Threads.end();
    auto thread_data_iter = std::find_if(threads_begin, threads_end, [thread_handle](const ThreadData& thread_data) { return thread_data.Handle == thread_handle; });

    if(thread_data_iter == threads_end)
    {
        m_Threads.push_back(ThreadData{ thread_handle });
        threads_begin = m_Threads.begin();
        threads_end = m_Threads.end();
        thread_data_iter = threads_end - 1;
    }

    auto& events = thread_data_iter->Events;
    auto cur_idx = events.size();
    EventId result{ static_cast<uint32_t>(thread_data_iter - threads_begin), static_cast<uint32_t>(cur_idx) };

    events.emplace_back(name, m_Timer.time());
    return result;
}

void Profiler::endCPUEvent(EventId evt_id)
{
    std::lock_guard<std::mutex> lock(m_Mutex);

    if(m_Status & DISCARD_EVENT)
        return;

    // basically you need this if you use external profilers to insert label automatically

    m_Threads[evt_id.ThreadIndex].Events[evt_id.EventIndex].m_EndTime = m_Timer.time();
}
}