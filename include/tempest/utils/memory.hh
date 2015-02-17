/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2010-2014 Zdravko Velinov
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

#ifndef _TEMPEST_MEMORY_HH
#define _TEMPEST_MEMORY_HH

#include <new>
#include <limits>
#include <cstddef>
#include <memory>

#ifdef _WIN32
#   ifdef __MINGW32__
#       define _aligned_malloc __mingw_aligned_malloc
#       define _aligned_free __mingw_aligned_free
#   endif
#   include <malloc.h>
#   define __ALIGNED_ALLOC(ptr, nbytes, align) ptr = _aligned_malloc(nbytes, align)
#   define __ALIGNED_DEALLOC(ptr) _aligned_free(ptr)
#elif defined(LINUX)
#   include <stdlib.h>
#   define  __ALIGNED_ALLOC(ptr, nbytes, align) posix_memalign(&ptr, align, nbytes)
#   define  __ALIGNED_DEALLOC(ptr) free(ptr)
#else
#   error "Unsupported platform"
#endif

#ifdef _MSC_VER
#   define TGE_FORTGE_ALIGNMENT
#else
//! Use this keyword to force alignment on callbacks and similar functions that may come from API that does not enforce proper alignment.
#   define TGE_FORTGE_ALIGNMENT __attribute__((force_align_arg_pointer))
#endif

#include "tempest/utils/patterns.hh"

#ifdef _WIN32
#   include <malloc.h>
#   define TGE_ALLOCA _alloca
#else
#   include <alloca.h>
#   define TGE_ALLOCA alloca
#endif

namespace Tempest
{
/*! \brief Allocator used for allocating memory for objects within Tempest Engine.
 * 
 *  \remarks It is not intended to be used directly. Refer to TGE_ALLOCATE for more information about how to manually manage memory.
 *  \tparam T         the type of the object that is being allocated.
 *  \tparam alignment memory alignment size in bytes.
 */
template<class T, int aligment>
struct ObjectAllocator
{
    //! Allocates a single object the specified type with the specified alignment.
    static inline T* allocate()
    {
        return static_cast<T*>(::operator new(sizeof(T)));
    }

    //! Deallocates a single object of the specified type.
    static inline void deallocate(T* ptr)
    {
        delete ptr;
    }
};

//! \brief Specialization for 16 bytes aligned objects, e.g. SSE.
template<class T>
struct ObjectAllocator<T, 16>
{
    //! Allocates a single object the specified type with the specified alignment.
    static inline T* allocate()
    {
        void* ptr;
        __ALIGNED_ALLOC(ptr, sizeof(T), 16);
        return reinterpret_cast<T*>(ptr);
    }

    //! Deallocates a single object of the specified type.
    static inline void deallocate(T* ptr)
    {
        ptr->~T();
        __ALIGNED_DEALLOC(ptr);
    }
};

/*! \brief Wraps the deallocator used within Tempest Engine.
 *  \remarks It should not be used directly. Refer to TGE_DEALLOCATE for more information how to deallocate memory.
 */
template<class T>
inline void DeallocFunction(T* p)
{
    ObjectAllocator<T, std::alignment_of<T>::value>::deallocate(p);
}

/*! \brief Internal implementation of TGE_ALLOCATE.
 *  \remarks  Array alAST::Location is discouraged; use DynamicArray instead.
 */
#define _TGE_ALLOCATE(type) new (Tempest::ObjectAllocator<type, std::alignment_of<type>::value>::allocate()) type

/*! \brief Internal implementation of TGE_DEALLOCATE.
 *  \remarks Potentially dangerous, if used for abstract data types, and it won't compile on some compilers, anyway.
 */
#define _TGE_DEALLOCATE(p) Tempest::DeallocFunction(p)

#ifdef TGE_MEMORY_DEBUG
class MemoryDebugger: public Singleton<MemoryDebugger>
{
    typedef std::unordered_map<void*, string> PointerMap;
    PointerMap          m_Allocated,
                        m_Deallocated;
public:
    MemoryDebugger();
    ~MemoryDebugger();
    
    template<class T>
    static T* allocate(const string& info)
    {
        T* ptr = Tempest::ObjectAllocator<T, alignment_of<T>::value>::allocate();
        MemoryDebugger::getSingleton().registerPointer(ptr, info);
        return ptr;
    }
    
    template<class T>
    static void deallocate(T* ptr, const string& info)
    {
        if(ptr && MemoryDebugger::getSingleton().unregisterPointer(ptr, info))
            _TGE_DEALLOCATE(ptr);
    }
    
    bool isAllocated(void* ptr);
    string getAllocatedInfo(void* ptr);
    string getDeallocatedInfo(void* ptr);
private:
    bool registerPointer(void* ptr, const string& info);
    bool unregisterPointer(void* ptr, const string& info);
};
#endif

#ifndef TGE_MEMORY_DEBUG
#   define TGE_ALLOCATE(type) _TGE_ALLOCATE(type)
#   define TGE_DEALLOCATE(p) _TGE_DEALLOCATE(p)
#else
#   define TGE_ALLOCATE(type) new (Tempest::MemoryDebugger::allocate<type>(__FILE__ + std::string(":") + TO_STRING(__LINE__))) type
#   define TGE_DEALLOCATE(p) Tempest::MemoryDebugger::deallocate(p, __FILE__ + std::string(":") + TO_STRING(__LINE__))
#endif

template<class T>
class AlignedSTLAllocator
{
public:
    typedef T*        pointer;
    typedef const T*  const_pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef T         value_type;
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    T* address(T& r) const
    {
        return &r;
    }

    const T* address(const T& s) const
    {
        return &s;
    }

    size_t max_size() const
    {
        return std::numeric_limits<size_t>::max()/sizeof(T);
    }

    template <typename U>
    struct rebind
    {
        typedef AlignedSTLAllocator<U> other;
    };

    bool operator!=(const AlignedSTLAllocator& other) const
    {
        return !(*this == other);
    }

    template<typename... TArgs>
    void construct(pointer p, TArgs&&... args)
    {
        ::new(reinterpret_cast<void *>(p)) T(std::forward<TArgs>(args)...);
    }


    void destroy(T* const p) const { p->~T(); }

    bool operator==(const AlignedSTLAllocator& other) const
    {
        return true;
    }

    AlignedSTLAllocator() {}
    AlignedSTLAllocator(const AlignedSTLAllocator&) {}
    template <typename U> AlignedSTLAllocator(const AlignedSTLAllocator<U>&) { }
    ~AlignedSTLAllocator() {}

    T* allocate(size_t n) const
    {
        void* ptr;
        __ALIGNED_ALLOC(ptr, sizeof(T)*n, 16);
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* const ptr, size_t n) const
    {
        __ALIGNED_DEALLOC(ptr);
    }
private:
    AlignedSTLAllocator& operator=(const AlignedSTLAllocator&);
};

template<class T>
struct Deleter
{
        void operator()(T* p) const
        {
                TGE_DEALLOCATE(p);
        }
};

template<class T>
inline T AlignAddress(T addr, T alignment) { return (addr + alignment - 1) & ~(alignment - 1); }

template<typename T, typename... TArgs>
inline std::shared_ptr<T> make_aligned_shared(TArgs&&... args)
{
    return std::allocate_shared<T>(AlignedSTLAllocator<T>(), std::forward<TArgs>(args)...);
}

template<typename T, typename... TArgs>
inline std::unique_ptr<T, Deleter<T>> make_aligned_unique(TArgs&&... args)
{
    return std::unique_ptr<T, Deleter<T>>(TGE_ALLOCATE(T)(std::forward<TArgs>(args)...), Deleter<T>());
}

template<typename T, typename... TArgs>
inline std::unique_ptr<T> make_unique(TArgs&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<TArgs>(args)...));
}
}

#endif /* _TEMPEST_MEMORY_HH */
