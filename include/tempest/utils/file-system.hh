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

#ifndef _TEMPEST_FILESYSTEM_HH_
#define _TEMPEST_FILESYSTEM_HH_

#include "tempest/utils/types.hh"

#include <list>
#include <vector>
#include <unordered_map>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN 1
#   include <windows.h>
#elif defined(LINUX)
#   include <sys/epoll.h>
#else
#	error "Unsupported platform"
#endif

namespace Tempest
{
//! Represents a path to file system object.
class Path
{
    //! The string that contains the actual path in native file format.
    string m_Path;
public:
    //! Default constructor.
    explicit Path();

    //! Constructor.
    explicit Path(const string& path);

    //! Copy constructor.
    Path(const Path& path);

    //! Assignment operator.
    Path& operator=(const Path& path);
    
    //! Sets the path.
    void set(const string& path);

    //! Returns the current path.
    string get() const;

    //! Returns just the file name.
    string filename() const;
    
    //! Returns the file name without extension.
    string filenameWOExt() const;

    //! Extracts a relative path to the specified one.
    string relativePath(const Path& p) const;

    //! Extracts just the directory part of the path.
    string directoryPath() const;

    //! Returns whether the object contains a valid path.
    bool isValid() const;

    //! Returns whether the path is pointing to a directory.
    bool isDirectory() const;
};

//! Equal to operator.
bool operator==(const Path& lhs, const Path& rhs);

//! Represents the result of file system operation.
enum FSOpResult
{
    TGE_FS_NO_ERROR, //!< The function has completed its task successfully.
    TGE_FS_EXISTS,   //!< The directory already exists.
    TGE_FS_ERROR     //!< An error has occurred while executing the specified function.
};

/*! \brief Represents a file system directory
 *  
 *  This is the preferred way to access file system directories than using
 *  the native file system APIs
 */
class Directory
{
    //! The name of the actual file system node.
    Path                m_DirNode;
    //! Directory's children.
    std::list<Path>     m_Nodes;
public:
    //! The iterator type used for traversing the directory children.
    typedef std::list<Path>::const_iterator iterator;

    //! Default constructor.
    explicit Directory();
    
    /*! \brief Constructor.
     *  \param path     the path to this directory.
     */
    explicit Directory(const Path& path);

    /*! \brief Opens the specified directory.
     *  \param path     the new path that must be used for this directory.
     */
    bool open(const Path& path);
    
    //! Returns the first child of this directory.
    iterator begin() const;
    
    //! Returns the end of the list of children.
    iterator end() const;

    //! Returns whether the path pointed by this object is a valid directory.
    bool isValid() const;

    //! Returns the path to this directory.
    Path getPath() const;
    
    //! Encapsulates the native function for creating a new directory.
    static FSOpResult mkdir(const string& name);
    
    //! Encapsulates the native function for deleting a directory.
    static FSOpResult rmdir(const string& name);
};

//! Describes a file system event, such as file modification, deletion, creation.
enum FSEventType
{
   TGE_FS_EVENT_UNKNOWN     = 0,      //!< Unknown file system event.
   TGE_FS_EVENT_MODIFY      = 1,      //!< The monitored file/directory has been modified.
   TGE_FS_EVENT_CREATE      = 1 << 1, //!< A new file was created within the monitored directory.
   TGE_FS_EVENT_DELETE      = 1 << 2, //!< A file was deleted within the monitored directory.
   TGE_FS_EVENT_DELETE_SELF = 1 << 3, //!< The monitored file/directory was deleted.
   TGE_FS_EVENT_MOVED_TO    = 1 << 4, //!< A file within the monitored directory was moved to another location.
   TGE_FS_EVENT_MOVED_FROM  = 1 << 5, //!< A file was moved to the monitored directory from another location.
   TGE_FS_EVENT_MOVE_SELF   = 1 << 6  //!< The monitored file/directory was moved to another location.
};

struct FSEvent
{
    int         type;
    string      name;
};

typedef std::vector<FSEvent> FSEvents;

/*! \brief Collects file system events and provides them in a cross platform fashion
 *
 *  It is used for monitoring changes of files in particular set of directories. That
 *  sort of functionality is used for constructing resource managers and file indexers.
 */
class FSPollingService
{
#ifdef _WIN32
    typedef size_t HandleType;

    struct MonitoredDirectory
    {
        string		name;
        HANDLE		handle;
        char		buffer[16384];
        OVERLAPPED  overlapped;

        MonitoredDirectory(string name);
            ~MonitoredDirectory();

        void restartMonitoring();

        bool operator==(const string& val) { return name == val; }
    };

    typedef std::unique_ptr<MonitoredDirectory, Deleter<MonitoredDirectory>> MonitoredDirectoryPtr;

    typedef std::vector<MonitoredDirectoryPtr> MonitoredDirectories;
    MonitoredDirectories	  m_Handles;
    HANDLE					  m_CompletionPort;
#elif defined(LINUX)
    //! File descriptor used for accessing inotify.
    int                       m_FD;
    //! epoll-related file descriptor for polling events received from inotify.
    int                       m_EPollFD;

    //! Index to the last element of incomplete event.
    size_t                    m_BufferIdx;
    //! Used for extracting information about epoll events.
    epoll_event               m_Events[32];
    //! Used for extracting the real events captured by inotify.
    char                      m_Buffer[16384];

    typedef int	HandleType;

    //! The type of the map between the monitored paths and their respective identifiers.
    typedef std::unordered_map<string, HandleType> WatchMap;
    //! Map between the monitored paths and their respective identifiers.
    WatchMap                  m_Watches;
#endif
public:
    /*! \brief Constructor.
     *
     *  Initializes some default values. Call initPollingService for complete initialization.
     */
    FSPollingService();
    //! Destructor.
     ~FSPollingService();

    /*! Initialize some basic resources.
     * 
     *  \returns Returns false, if there was failure during the initialization.
     */
    bool initPollingService();
        
    /*! \brief Add directory to the monitored set.
     *
     *  Monitores all files within the specified directory for changes. It does not support recursion.
     *  You may use this function to register directory that you are going to monitor by calling FSPollingService::poll. 
     *
     *  \param dir      the directory which is going to be monitored for changes.
     */
    bool addWatch(const Directory& dir);

    /* \brief Add directory to the monitored set.
        *
        *  Monitores all files within the specified directory for changes. It does not support recursion.
        *  You may use this function to register directory that you are going to monitor by calling FSPollingService::poll. 
        *
        *  \param path		the path to the directory which is going to be monitored for changes.
        */
    bool addWatch(const Path& path);

    /*! \brief Remove directory from the monitored set.
     *
     *  Removes a directory from the monitored set. You are not going to receive events about any associated files or
     *  directories afterwards if they are not registered within this object's monitored set.
     *
     *  \param dir		the directory which is no longer going to be monitored for changes.
     */
    bool removeWatch(const Directory& dir);

    /*! \brief Remove directory from the monitored set.
     *
     *  Removes a directory from the monitored set. You are not going to receive events about any associated files or
     *  directories afterwards if they are not registered within this object's monitored set.
     *
     *  \param path		the path to the directory which is no longer going to be monitored for changes.
     */
    bool removeWatch(const Path& path);
        
    /*! \brief Extract all received events.
     *
     *  Collects all received events by this object and puts them in a single list. For more information about all possible
     *  events refer to FSEventType. The events might be related to changes of the monitored directory and its
     *  immediate children, e.g. there is not any support for recursion.
     *
     *  \param evts		the list of the received events.
     */
    bool poll(FSEvents& evts);
};
}

#endif /* _TEMPEST_FILESYSTEM_HH_ */
