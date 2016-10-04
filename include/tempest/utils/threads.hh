/*   The MIT License
 *
 *   Tempest Engine
 *   Copyright (c) 2015 Zdravko Velinov
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

#ifndef _TEMPEST_THREADS_HH_
#define _TEMPEST_THREADS_HH_

#include <list>
#include <thread>
#include <memory>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <condition_variable>

#include "tempest/utils/system.hh"

namespace Tempest
{
class ThreadPool;

class Task
{
    friend class ThreadPool;
protected:
    enum class ExecutionPattern: uint8_t
    {
        SingleShot,
        SplitWork
    };
    
    ExecutionPattern m_ExecutionPattern = ExecutionPattern::SingleShot;
    uint64_t         m_ScheduledThreads = 0;
    uint64_t         m_CompletedThreads = 0;
public:
    virtual ~Task()=default;

    bool canSchedule(uint32_t thread_num)
    {
        return m_ScheduledThreads == 0 ||                                             // Regular threads
               (m_CompletedThreads != m_ScheduledThreads &&                           // or ones that probably have work to be executed
                (m_CompletedThreads & (1ULL << (uint64_t)thread_num)) == 0 &&         // and obviously are not finished
                m_ExecutionPattern == Task::ExecutionPattern::SplitWork);
    }

    bool isComplete()
    {
        return m_ScheduledThreads == m_CompletedThreads && m_CompletedThreads;
    }

    virtual void execute(uint32_t worker_id) = 0;
};

template<class TExec>
class ParallelForLoop: public Task
{
    std::atomic_uint m_Counter;
    uint32_t         m_TotalCount;
    uint32_t         m_ChunkSize;
    TExec            m_Execution;

public:
    ParallelForLoop(uint32_t total_count, uint32_t chunk_size, TExec execution)
        :   m_TotalCount(total_count),
            m_ChunkSize(chunk_size),
            m_Execution(execution)
	{
		m_Counter.store(0);
		m_ExecutionPattern = Task::ExecutionPattern::SplitWork;
	}

    ParallelForLoop(ParallelForLoop&& loop)
        :   m_TotalCount(loop.m_TotalCount),
            m_ChunkSize(loop.m_ChunkSize),
            m_Execution(std::move(m_Execution))
    {
        m_Counter.store(0);
        m_ExecutionPattern = Task::ExecutionPattern::SplitWork;
    }

	void setTotalCount(uint32_t total_count) { m_TotalCount = total_count; }
	void setChunkSize(uint32_t chunk_size) { m_ChunkSize = chunk_size; }

    void reset(uint32_t total_count)
    {
        TGE_ASSERT(m_ScheduledThreads == m_CompletedThreads && m_ScheduledThreads, "Reset before completion");
        m_Counter.store(0);
        m_ScheduledThreads = 0;
        m_CompletedThreads = 0;
    }

    virtual void execute(uint32_t worker_id) override
    {
        for(uint32_t p; (p = m_Counter.fetch_add(m_ChunkSize)) < m_TotalCount;)
        {
            m_Execution(worker_id, p, m_ChunkSize);
        }
    }
};

template<class TExec>
ParallelForLoop<TExec> CreateParallelForLoop(uint32_t total_count, uint32_t chunk_size, TExec execution)
{
    return ParallelForLoop<TExec>(total_count, chunk_size, execution);
}

template<class TExec>
class ParallelForLoop2D: public Task
{
    std::atomic_uint m_Counter;
    uint32_t         m_Width;
    uint32_t         m_TotalCount;
    uint32_t         m_ChunkSize;
    TExec            m_Execution;

public:
    ParallelForLoop2D(uint32_t width, uint32_t total_count, uint32_t chunk_size, TExec execution)
        :   m_Width(width),
            m_TotalCount(total_count),
            m_ChunkSize(chunk_size),
            m_Execution(execution)
	{
		m_Counter.store(0);
		m_ExecutionPattern = Task::ExecutionPattern::SplitWork;
	}

    ParallelForLoop2D(ParallelForLoop2D&& loop)
        :   m_Width(loop.m_Width),
            m_TotalCount(loop.m_TotalCount),
            m_ChunkSize(loop.m_ChunkSize),
            m_Execution(std::move(m_Execution))
    {
        m_Counter.store(0);
        m_ExecutionPattern = Task::ExecutionPattern::SplitWork;
    }

	void setChunkSize(uint32_t chunk_size) { m_ChunkSize = chunk_size; }

    void reset(uint32_t total_count)
    {
        TGE_ASSERT(m_ScheduledThreads == m_CompletedThreads && m_ScheduledThreads, "Reset before completion");
        m_Counter.store(0);
        m_ScheduledThreads = 0;
        m_CompletedThreads = 0;
    }

    virtual void execute(uint32_t worker_id) override
    {
        auto chunk_size = m_ChunkSize;
        for(uint32_t idx; (idx = m_Counter.fetch_add(chunk_size)) < m_TotalCount;)
        {
            for(uint32_t chunk_end = std::min(idx + chunk_size, (uint32_t)m_TotalCount); idx < chunk_end; ++idx)
            {
                uint32_t x = idx % m_Width;
                uint32_t y = idx / m_Width;
                m_Execution(worker_id, x, y);
            }
        }
    }
};


template<class TLoopBody>
ParallelForLoop2D<TLoopBody> CreateParallelForLoop2D(uint32_t width, uint32_t height, uint32_t chunk_size, TLoopBody body)
{
    auto area = width*height;
    return ParallelForLoop2D<TLoopBody>(width, area, chunk_size, body);
}

class ThreadPool
{
    bool                              m_Active = true;

    uint32_t                          m_ExternalThreads = 0;

    std::mutex                        m_TaskConsumeMutex;
    std::condition_variable           m_WakeUpCondition;
    std::condition_variable		      m_WakeUpMain;
    std::list<Task*>                  m_Tasks;

    std::unique_ptr<std::thread[]>    m_WorkerThreads;
    uint32_t                          m_WorkerThreadCount;
public:
    ThreadPool(uint32_t thread_count = std::max(1u, System::GetNumberOfProcessors()));
     ~ThreadPool();

    uint32_t getWorkerThreadCount() const { return m_WorkerThreadCount; }
    uint32_t getThreadCount() const { return m_WorkerThreadCount + m_ExternalThreads; }

    // This is if you want to help from external threads
    uint32_t allocateThreadNumber()
    {
        std::lock_guard<std::mutex> lock(m_TaskConsumeMutex);
        return m_WorkerThreadCount + (m_ExternalThreads++);
    }

    void enqueueTask(Task* task);

    void help(uint32_t thread_num, Task* task);
    void waitAndHelp(uint32_t thread_num, Task* task);
    void wait(uint32_t thread_num, Task* task);
    void shutdown();

private:
    void consumeTasks(uint32_t thread_num);
    bool consumeSingleTask(std::unique_lock<std::mutex>& lock, uint32_t thread_num);
};
}

#endif // _TEMPEST_THREADS_HH_
