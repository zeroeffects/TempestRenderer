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

#include "tempest/utils/threads.hh"
#include "tempest/utils/system.hh"

#include <sstream>

namespace Tempest
{
ThreadPool::ThreadPool(uint32_t thread_count)
    :   m_WorkerThreads(new std::thread[thread_count]),
		m_WorkerThreadCount(thread_count) 
{
    for(uint32_t worker = 0; worker < m_WorkerThreadCount; ++worker)
    {
        auto& cur_thread = m_WorkerThreads[worker] = std::thread([this, worker](){ this->consumeTasks(worker); });
        std::stringstream ss;
        ss << "Worker Thread #" << worker + 1;
        auto name = ss.str();

        System::SetThreadName(cur_thread.native_handle(), name.c_str());
    }   
}

ThreadPool::~ThreadPool()
{
    if(m_WorkerThreads[0].joinable())
        shutdown();
}

void ThreadPool::enqueueTask(Task* task)
{
	{
	std::lock_guard<std::mutex> lock(m_TaskConsumeMutex);
	TGE_ASSERT(task->m_CompletedThreads == 0 && task->m_ScheduledThreads == 0, "Invalid task");
	m_Tasks.push_back(task);
	}
	m_WakeUpCondition.notify_all();
}

bool ThreadPool::consumeSingleTask(std::unique_lock<std::mutex>& lock, uint32_t thread_num)
{
    auto thread_bit = (1ULL << (uint64_t)thread_num);

    Task* task = nullptr;
    {
        auto iter = m_Tasks.begin(), iter_end = m_Tasks.end();
        for(; iter != iter_end; ++iter)
        {
            if((*iter)->canSchedule(thread_num))
                break;
        }

        if(iter == iter_end)
        {
            return false;
        }

		task = *iter;
        task->m_ScheduledThreads |= thread_bit;
        if(task->m_ExecutionPattern == Task::ExecutionPattern::SingleShot)
        {
            m_Tasks.erase(iter);
        }
    }

    {
        lock.unlock();
        auto auto_lock = CreateAtScopeExit([&lock]() { lock.lock(); });
        task->execute(thread_num);
    }

    {
        TGE_ASSERT((task->m_CompletedThreads & thread_bit) == 0, "Invalid completion");
        task->m_CompletedThreads |= thread_bit;

        if(task->m_ExecutionPattern == Task::ExecutionPattern::SingleShot)
		{
			TGE_ASSERT(task->isComplete(), "Invalid completion of one shot task");
            m_WakeUpMain.notify_all();
		}
		else if(task->isComplete())
        {
            auto iter = std::find(m_Tasks.begin(), m_Tasks.end(), task);
            TGE_ASSERT(iter != m_Tasks.end(), "Premature completion");
            m_Tasks.erase(iter);
            m_WakeUpMain.notify_all();
        }
    }

    return true;
}

void ThreadPool::consumeTasks(uint32_t thread_num)
{
    for(;;)
    {
        std::unique_lock<std::mutex> lock(m_TaskConsumeMutex);
        if(consumeSingleTask(lock, thread_num))
            continue;
        if(!m_Active)
            break;
        m_WakeUpCondition.wait(lock);
    }
}

void ThreadPool::help(uint32_t thread_num, Task* task)
{
    std::unique_lock<std::mutex> lock(m_TaskConsumeMutex);
    consumeSingleTask(lock, thread_num);
}

void ThreadPool::waitAndHelp(uint32_t thread_num, Task* task)
{
    std::unique_lock<std::mutex> lock(m_TaskConsumeMutex);
	while(!task->isComplete() && task->canSchedule(thread_num))
    {
		consumeSingleTask(lock, thread_num);
    }
	// Avoid starvation
	if(!task->isComplete())
	{
		m_WakeUpMain.wait(lock, [task]() { return task->isComplete(); });
	}
}

void ThreadPool::wait(uint32_t thread_num, Task* task)
{
    std::unique_lock<std::mutex> lock(m_TaskConsumeMutex);
	if(!task->isComplete())
	{
		m_WakeUpMain.wait(lock, [task]() { return task->isComplete(); });
	}
}

void ThreadPool::shutdown()
{
    {
    std::lock_guard<std::mutex> lock(m_TaskConsumeMutex);
    m_Active = false;
    }
    m_WakeUpCondition.notify_all();
    for(size_t i = 0; i < m_WorkerThreadCount; ++i)
    {
        m_WorkerThreads[i].join();
    }
}
}