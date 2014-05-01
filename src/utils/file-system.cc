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

#include "tempest/utils/file-system.hh"

#include "tempest/utils/logging.hh"

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN 1
#   include <winerror.h>
#   ifdef __MINGW32__
#       define off64_t _off64_t
#       define ERROR_ABANDONED_WAIT_0 735L
#   endif
#   include <windows.h>
#   include <direct.h>
#elif defined(LINUX)
#   include <unistd.h>
#   include <dirent.h>
#   include <sys/stat.h>
#   include <sys/types.h>
#   include <sys/epoll.h>
#   include <sys/inotify.h>
#   include <sys/fcntl.h>
#else
#	error "Unsupported platform"
#endif

#include <cassert>
#include <cstring>
#include <algorithm>


namespace Tempest
{
Path::Path()
{
}

Path::Path(const string& path)
{
    this->set(path);
}

Path::Path(const Path& path)
    :   m_Path(path.m_Path) {}

Path& Path::operator=(const Path& path)
{
    m_Path = path.m_Path;
    return *this;
}

bool operator==(const Path& lhs, const Path& rhs)
{
    return lhs.get() == rhs.get();
}

#define PATH_DELIM '/'
#ifdef _WIN32
#   define INVALID_PATH_DELIM '/'
#elif defined(LINUX)
#   define INVALID_PATH_DELIM '\\'
#else
#   error "Unsupported platform"
#endif

void Path::set(const string& path)
{
    m_Path = path;
    for(size_t i = 0; i < m_Path.size(); ++i)
    {
        if(m_Path[i] == INVALID_PATH_DELIM)
            m_Path[i] = PATH_DELIM;
    }
}

string Path::get() const
{
    return m_Path;
}

string Path::relativePath(const Path& p) const
{
    if(m_Path.size() >= p.m_Path.size() ||
       p.m_Path.compare(0, m_Path.size(), m_Path) != 0)
        return string();
    size_t sep = m_Path.size();
    if(p.m_Path[sep] == PATH_DELIM)
        ++sep;
    return p.m_Path.substr(sep);
}

string Path::directoryPath() const
{
    size_t idx = m_Path.find_last_of(PATH_DELIM);
    return idx != string::npos ? m_Path.substr(0, idx) : "";
}

string Path::filename() const
{
    size_t idx = m_Path.find_last_of(PATH_DELIM);
    return idx != string::npos ? m_Path.substr(idx+1) : m_Path;
}

string Path::filenameWOExt() const
{
    size_t first_char = m_Path.find_last_of(PATH_DELIM);
    first_char = first_char == string::npos ? 0 : first_char + 1;
    size_t last_char = string::npos;
    for(size_t i = m_Path.size()-1; i >= first_char; ++i)
        if(m_Path[i] == '.')
        {
            last_char = i;
            break;
        }
    return m_Path.substr(first_char, last_char);
}

#ifdef _WIN32
bool Path::isValid() const
{
    if(m_Path.empty())
        return false;
    return GetFileAttributes(m_Path.c_str()) != INVALID_FILE_ATTRIBUTES;
}
#elif defined(LINUX)
bool Path::isValid() const
{
    struct stat statbuf;
    return stat(m_Path.c_str(), &statbuf) == 0;
}
#else
#   error "Unsupported platform"
#endif


#ifdef _WIN32
bool Path::isDirectory() const
{
    DWORD result = GetFileAttributes(m_Path.c_str());
    if(result == INVALID_FILE_ATTRIBUTES)
        return false;
    return (result & FILE_ATTRIBUTE_DIRECTORY) != 0;
}
#else
bool Path::isDirectory() const
{
    struct stat statbuf;
    if(stat(m_Path.c_str(), &statbuf) < 0)
        return false;
    return (statbuf.st_mode & S_IFDIR) != 0;
}
#endif

Directory::Directory() {}

Directory::Directory(const Path& node)
{
    this->open(node);
}

Directory::iterator Directory::begin() const
{
    return m_Nodes.begin();
}

Directory::iterator Directory::end() const
{
    return m_Nodes.end();
}

#ifdef _WIN32
bool Directory::open(const Path& path)
{
    if(!path.isDirectory())
        return false;

    string path_str = path.get() + "\\*";
    WIN32_FIND_DATA find_data;
    HANDLE hnd = FindFirstFile(path_str.c_str(), &find_data);
    if(hnd == INVALID_HANDLE_VALUE)
        return false;

    do {
        if(strcmp(find_data.cFileName, ".") && strcmp(find_data.cFileName, ".."))
            m_Nodes.push_back(Path(path.get() + "\\" + find_data.cFileName));
    } while(FindNextFile(hnd, &find_data));
    
    if(GetLastError() != ERROR_NO_MORE_FILES)
    {
        m_Nodes.clear();
        return false;
    }

    FindClose(hnd);
    m_DirNode = path;
    return true;
}
#elif defined(LINUX)
bool Directory::open(const Path& path)
{
    struct dirent* p_dirent;
    DIR* dir = opendir(path.get().c_str());
    if(!dir)
        return false;
    while((p_dirent = readdir(dir)) != 0)
        if(strcmp(p_dirent->d_name, ".") && strcmp(p_dirent->d_name, ".."))
            m_Nodes.push_back(Path(path.get() + "/" + p_dirent->d_name));
    closedir(dir);
    m_DirNode = path;
    return true;
}
#else
#   error "Unsupported platform"
#endif

bool Directory::isValid() const
{
    return m_DirNode.isValid();
}

Path Directory::getPath() const
{
    return m_DirNode;
}

FSOpResult Directory::mkdir(const string& name)
{
#ifdef _WIN32
    int res = ::_mkdir(name.c_str());
#elif defined(LINUX)
    int res = ::mkdir(name.c_str(), 0755);
#else
#   error "Unsupported platform"
#endif
    if(res < 0)
        return errno == EEXIST ? TGE_FS_EXISTS : TGE_FS_ERROR;
    return TGE_FS_NO_ERROR;
}

FSOpResult Directory::rmdir(const string& name)
{
#ifdef _WIN32
    int res = ::_rmdir(name.c_str());
#elif defined(LINUX)
    int res = ::rmdir(name.c_str());
#else
#   error "Unsupported platform"
#endif
    if(res < 0)
        return TGE_FS_ERROR;
    return TGE_FS_NO_ERROR;
}

#ifdef _WIN32
extern string GetLastErrorString();

FSPollingService::MonitoredDirectory::MonitoredDirectory(string _name)
    :	name(_name),
        handle(::CreateFile(name.c_str(),
                            FILE_LIST_DIRECTORY,
                            FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                            nullptr,
                            OPEN_EXISTING,
                            FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OVERLAPPED,
                            nullptr))
{
	if(handle == INVALID_HANDLE_VALUE)
        THROW_EXCEPTION("The application has failed to set up a file system handle for the following directory: " + _name + ": " + GetLastErrorString());
}

FSPollingService::MonitoredDirectory::~MonitoredDirectory()
{
    CancelIo(handle);
    CloseHandle(handle);
}

void FSPollingService::MonitoredDirectory::restartMonitoring()
{
    ZeroMemory(&overlapped, sizeof(OVERLAPPED));
    DWORD bytes = 0;
    auto ret = ::ReadDirectoryChangesW(handle,
                                        buffer,
                                        TGE_FIXED_ARRAY_SIZE(buffer),
                                        TRUE,
                                        FILE_NOTIFY_CHANGE_FILE_NAME|
                                        FILE_NOTIFY_CHANGE_DIR_NAME|
                                        FILE_NOTIFY_CHANGE_LAST_WRITE,
                                        &bytes,
                                        &overlapped,
                                        nullptr);
    if(ret == FALSE)
        THROW_EXCEPTION("The application has failed to enqueue directory change listening operation: " + GetLastErrorString());
}

FSPollingService::FSPollingService()
	:	m_CompletionPort(nullptr)
{
}

FSPollingService::~FSPollingService()
{
}

void FSPollingService::addWatch(const Directory& dir)
{
    addWatch(dir.getPath());
}

VOID  NotificationCompletion(DWORD  , DWORD  , LPVOID ) {}

void FSPollingService::addWatch(const Path& path)
{
    auto str = path.get();
    MonitoredDirectoryPtr ptr = make_aligned_unique<MonitoredDirectory>(path.get());
    m_CompletionPort = CreateIoCompletionPort(ptr->handle, m_CompletionPort, reinterpret_cast<ULONG_PTR>(ptr.get()), 0);
    if(!m_CompletionPort)
        THROW_EXCEPTION("The application has failed to set up a completion port: " + path.get() + ": " + GetLastErrorString());

    ptr->restartMonitoring();
        
    m_Handles.push_back(std::move(ptr));
}

void FSPollingService::removeWatch(const Directory& dir)
{
    removeWatch(dir.getPath());
}

void FSPollingService::removeWatch(const Path& path)
{
	auto iter = std::find_if(m_Handles.begin(), m_Handles.end(), [&path](const MonitoredDirectoryPtr& dir) { return dir->name == path.get(); });
    if(iter == m_Handles.end())
        THROW_EXCEPTION("Unmanaged file system watch: " + path.get());
	m_Handles.erase(iter);
}

void FSPollingService::poll(FSEvents& evts)
{
    evts.clear();
    DWORD bytes_transferred;
    MonitoredDirectory* key;
    LPOVERLAPPED overlapped;

    while(GetQueuedCompletionStatus(m_CompletionPort, &bytes_transferred, reinterpret_cast<PULONG_PTR>(&key), &overlapped, 0))
    {
        char* next = key->buffer;
        FILE_NOTIFY_INFORMATION* fni;

        do
        {
            fni = (FILE_NOTIFY_INFORMATION*)next;
            next = next + fni->NextEntryOffset;

            int size = WideCharToMultiByte(CP_UTF8, 0, fni->FileName, (int)fni->FileNameLength/sizeof(wchar_t), nullptr, 0, nullptr, nullptr);
            string str(size, 0);
            WideCharToMultiByte(CP_UTF8, 0, fni->FileName, (int)fni->FileNameLength/sizeof(wchar_t), &str[0], size, nullptr, nullptr);

            FSEvent fsevent;
            fsevent.type = 0;
            if(fni->Action == FILE_ACTION_ADDED)
                fsevent.type |= TGE_FS_EVENT_CREATE;
            if(fni->Action == FILE_ACTION_REMOVED)
                fsevent.type |= str == key->name ? TGE_FS_EVENT_DELETE_SELF : TGE_FS_EVENT_DELETE;
            if(fni->Action == FILE_ACTION_MODIFIED)
                fsevent.type |= TGE_FS_EVENT_MODIFY;
            if(fni->Action == FILE_ACTION_RENAMED_OLD_NAME)
            {
                TGE_ASSERT(fni->NextEntryOffset, "There should be a next entry that contains the new name!");
                fni = (FILE_NOTIFY_INFORMATION*)next;
                next = next + fni->NextEntryOffset;
                TGE_ASSERT(fni->Action == FILE_ACTION_RENAMED_NEW_NAME, "New name expected after the older one!");
                fsevent.type |= TGE_FS_EVENT_MOVED_TO;
                // TODO: OMG ? Outside?
                size = WideCharToMultiByte(CP_UTF8, 0, fni->FileName, (int)fni->FileNameLength/sizeof(wchar_t), nullptr, 0, nullptr, nullptr);
                str.reserve(size);
                WideCharToMultiByte(CP_UTF8, 0, fni->FileName, (int)fni->FileNameLength/sizeof(wchar_t), &str[0], size, nullptr, nullptr);
            }
        
            // fsevent.type |= TGE_FS_EVENT_MOVED_FROM;
            // fsevent.type |= TGE_FS_EVENT_MOVE_SELF;
            fsevent.name = str;
            evts.push_back(fsevent);
        } while(fni->NextEntryOffset);
        key->restartMonitoring();
    }
    auto status = GetLastError();
    if(status != ERROR_ABANDONED_WAIT_0 && status != WAIT_TIMEOUT)
        THROW_EXCEPTION("The application has encountered an error while polling for file system events: " + GetLastErrorString());
    else
        return;
}
#elif defined(LINUX)
FSPollingService::FSPollingService()
    :   m_FD(-1),
        m_EPollFD(-1),
        m_BufferIdx(0)
{
}

FSPollingService::~FSPollingService()
{
    for(auto& pair : m_Watches)
        inotify_rm_watch(m_FD, pair.second);
    if(m_FD >= 0) close(m_FD);
}

bool FSPollingService::initPollingService()
{
    m_FD = inotify_init();
    if(m_FD < 0)
    {
        Log(LogLevel::Error, "The application has failed to initialize file system event polling service: ", strerror(errno));
        return false;
    }
    m_EPollFD = epoll_create(sizeof(m_FD));
    if(m_EPollFD < 0)
    {
        Log(LogLevel::Error, "The application has failed to initialize event polling service: ", strerror(errno));
        return false;
    }
    int flags = fcntl(m_FD, F_GETFL, 0);
    if(flags < 0)
    {
        Log(LogLevel::Error, "Failed to get fcntl flags: ", strerror(errno));
        return false;
    }
    int res = fcntl(m_FD, F_SETFL, flags|O_NONBLOCK);
    if(res < 0)
    {
        Log(LogLevel::Error, "Failed to set inotify descriptor as non-blocking: ", strerror(errno));
        return false;
    }
        
    struct epoll_event ev;
    ev.events = EPOLLIN|EPOLLOUT|EPOLLET;
    res = epoll_ctl(m_EPollFD, EPOLL_CTL_ADD, m_FD, &ev);
    if(res < 0)
    {
        Log(LogLevel::Error, "The application has failed to configure epoll: ", strerror(errno));
        return false;
    }
    return true;
}

bool FSPollingService::addWatch(const Directory& dir)
{
    return addWatch(dir.getPath());
}

bool FSPollingService::addWatch(const Path& path)
{
    auto str = path.get();
    int watch = inotify_add_watch(m_FD, str.c_str(), IN_MODIFY|IN_CREATE|IN_DELETE|IN_DELETE_SELF|IN_MOVED_TO|IN_MOVED_FROM|IN_MOVE_SELF);
    m_Watches[str.c_str()] = watch;
    if(watch == -1)
    {
        Log(LogLevel::Error, "The application has failed to set up a file system watch for the following file: ", path.get(), ": ", strerror(errno));
        return false;
    }
    return true;
}

bool FSPollingService::removeWatch(const Directory& dir)
{
    return removeWatch(dir.getPath());
}

bool FSPollingService::removeWatch(const Path& path)
{
    auto i = m_Watches.find(path.get());
    if(i == m_Watches.end())
    {
        Log(LogLevel::Error, "Unmanaged file system watch: ", path.get());
        return false;
    }
    inotify_rm_watch(m_FD, i->second);
    m_Watches.erase(i);
    return true;
}

bool FSPollingService::poll(FSEvents& evts)
{
    evts.clear();
    int events = epoll_wait(m_EPollFD, m_Events, TGE_FIXED_ARRAY_SIZE(m_Events), 0);
    if(events < 0)
    {
        Log(LogLevel::Error, "An error has occurred while polling for file system events: ", strerror(errno));
        return false;
    }
    if(events)
    {
        int n = 0;
        do
        {
            // It might seem as more than it is needed, but I don't plan on fixing bugs
            // in this code after several years just because there is sudden surge of events.
            n = read(m_FD, m_Buffer, TGE_FIXED_ARRAY_SIZE(m_Buffer) - m_BufferIdx);
            if(n < 0)
            {
                Log(LogLevel::Error, "An error has occurred while reading file system events: ", strerror(errno));
                return false;
            }
            size_t buf_end = n + m_BufferIdx, i = 0;
            for(m_BufferIdx = buf_end - i; i < buf_end; m_BufferIdx = buf_end - i)
            {
                if(m_BufferIdx < sizeof(inotify_event))
                {
                    break;
                }
                inotify_event* ev = reinterpret_cast<inotify_event*>(m_Buffer + i);
                i += ev->len + sizeof(inotify_event);
                if(buf_end < i)
                {
                    break;
                }
                FSEvent fsevent;
                fsevent.type = 0;
                if(ev->mask & IN_MODIFY)
                {
                    fsevent.type |= TGE_FS_EVENT_MODIFY;
                }
                if(ev->mask & IN_CREATE)
                {
                    fsevent.type |= TGE_FS_EVENT_CREATE;
                }
                if(ev->mask & IN_DELETE)
                    fsevent.type |= TGE_FS_EVENT_DELETE;
                if(ev->mask & IN_DELETE_SELF)
                {
                    fsevent.type |= TGE_FS_EVENT_DELETE_SELF;
                    // TODO: fixme
                }
                if(ev->mask & IN_MOVED_TO)
                {
                    fsevent.type |= TGE_FS_EVENT_MOVED_TO;
                }
                if(ev->mask & IN_MOVED_FROM)
                {
                    fsevent.type |= TGE_FS_EVENT_MOVED_FROM;
                }
                if(ev->mask & IN_MOVE_SELF)
                {
                    fsevent.type |= TGE_FS_EVENT_MOVE_SELF;
                }
                TGE_ASSERT(fsevent.type, "Unexpected compound event");
                fsevent.name = ev->name;
                
                evts.push_back(fsevent);
            }
            
            if(m_BufferIdx)
                std::copy_n(std::begin(m_Buffer) + i, m_BufferIdx, std::begin(m_Buffer));
        } while(n == TGE_FIXED_ARRAY_SIZE(m_Buffer));
    }
    return true;
}
#else
#   error "Unsupported platform"
#endif
}
