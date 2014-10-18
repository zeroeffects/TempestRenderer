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

#ifndef PATTERNS_HH_
#define PATTERNS_HH_

#include "tempest/utils/assert.hh"

#ifdef _MSC_VER
	#pragma warning(disable : 4200)
#endif

namespace Tempest
{
/*! \brief Common template for creating an object that has a single instance within the application.
 * 
 *  Singletons are objects that have just a single instance within the application and they are shared
 *  between multiple objects. Essentially, they serve the purpose of global variables with a little bit
 *  better memory management. In any case, you are discouraged to use them because they introduce hidden
 *  dependencies between objects. Also, it is awfully hard in a later stage to introduce multi-threading
 *  in your application when everything depends on a single object. That's mostly because locking a mutex
 *  is not for free, even if there are not any other objects accessing the memory.
 * 
 *  \tparam T   the type of the object that must have a single instance.
 */
template<class T>
class Singleton
{
    static T* m_Instance; /*! \brief Pointer to the single instance of the specified object type.
                           *  
                           *  Some people might look at this and say to themselves that it would be
                           *  much more clever to declare it in the global space in some file. The thing
                           *  is, that it turns the code into complete spaghetti at some point because
                           *  some people start to recognize it as legit pattern to carry around state.
                           */
public:
    //! Constructor.
    Singleton()
    {
        TGE_ASSERT(m_Instance == nullptr, "Singleton not initialized"); m_Instance = (T*)this;
    }

    //! Destructor.
     ~Singleton() { TGE_ASSERT(m_Instance, "Singleton not initialized or manually freed"); m_Instance = nullptr; }

    /*! \brief Gets a reference to the singleton object.
     *  \remarks You might as well wrap it behind some function. No one needs to know about our awful design.
     */
    static T& getSingleton() 
    { 
        TGE_ASSERT(m_Instance != nullptr, "Singleton not initialized at that time");
        return *m_Instance; 
    }
    
    /*! \brief Gets a pointer to the singleton object.
     *  \remarks You might as well wrap it behind some function. No one needs to know about our awful design.
     */
    
    static T* getSingletonPtr() { return m_Instance; }
};

// Convenience - calls all of your destructors, so you can pack arrays without being afraid that everything 
// is going to break appart.
template<class T>
struct PackedData
{
    const uint32    Count;
    T               Values[];
    PackedData(uint32 count)
        :   Count(count)
    {
        for(size_t i = 0; i < count; ++i)
        {
            new (&Values[i]) T;
        }
    }
    
    ~PackedData()
    {
        for(size_t i = 0, iend = Count; i < iend; ++i)
        {
            Values[i].~T();
        }
    }
};

template<class T, class... TArgs>
T* CreatePackedData(size_t count, TArgs&&... args)
{
    return new (malloc(sizeof(T) + count*sizeof(typename T::PackType))) T(count, args...);
}

template<class T>
void DestroyPackedData(T* ptr)
{
    ptr->~T();
    free(ptr);
}

#define PACKED_DATA(type) \
    typedef type PackType; \
    template<class T, class... TArgs> friend T* Tempest::CreatePackedData(size_t count, TArgs&&... args); \
    template<class T> friend void Tempest::DestroyPackedData(T* ptr); \
    Tempest::PackedData<type>  

template<class T> T* Singleton<T>::m_Instance = nullptr;

template<class T, class TDeleter> class ScopedObject;

template<class T, class TDeleter> 
class ScopedObject
{
    T                   m_Desc;
    TDeleter            m_Deleter;
public:
    ScopedObject() {}
    
    ScopedObject(T desc, TDeleter deleter)
        :   m_Desc(desc),
            m_Deleter(deleter) {}
    
    ScopedObject(TDeleter deleter)
        :   m_Deleter(deleter) {}
        
     ~ScopedObject() { if(m_Desc) m_Deleter(m_Desc); }

    ScopedObject& operator=(T t) { m_Desc = t; return *this; }

    T get() { return m_Desc; }
    const T get() const { return m_Desc; }
    
    T release() { auto tmp = m_Desc; m_Desc = T(); return tmp; }
};

template<class T, class TDeleter>
class ScopedObject<T*, TDeleter>
{
    T*                  m_Ptr;
    TDeleter            m_Deleter;
public:
    ScopedObject() {}
    
    ScopedObject(T* ptr, TDeleter deleter)
        :   m_Ptr(ptr),
            m_Deleter(deleter) {}
    
    ScopedObject(TDeleter deleter)
        :   m_Deleter(deleter) {}
        
     ~ScopedObject() { if(m_Ptr) m_Deleter(m_Ptr); }

    ScopedObject& operator=(T* t) { m_Ptr = t; return *this; }

    T* get() { return m_Ptr; }
    const T* get() const { return m_Ptr; }

    T release() { auto tmp = m_Ptr; m_Ptr = nullptr; return tmp; }
    
    T** operator&(){ return &m_Ptr; }
    operator T*() const { return m_Ptr; }
    T& operator*() { return *m_Ptr; }
    T* operator->() { return m_Ptr; }

    T* const * operator&() const { return &m_Ptr; }
    const T& operator*() const { return *m_Ptr; }
    const T* operator->() const { return m_Ptr; }
};

template<class TRollback>
class Transaction
{
    bool                   m_Status;
    TRollback              m_Rollback;
public:
    Transaction(TRollback _rollback)
        :   m_Status(true),
            m_Rollback(_rollback) {}
        
    ~Transaction() { if(m_Status) m_Rollback(); }
    
    void commit() { m_Status = false; }
};

template<class TFunc>
Transaction<TFunc> CreateTransaction(TFunc func)
{
    return Transaction<TFunc>(func);
}

template<class TFunc>
class AtScopeExit
{
    TFunc                  m_Func; 
public:
    AtScopeExit(TFunc func)
        :   m_Func(func) {}
    ~AtScopeExit() { m_Func(); }
};

template<class TFunc>
AtScopeExit<TFunc> CreateAtScopeExit(TFunc func)
{
    return AtScopeExit<TFunc>(func);
}

template<typename T, typename TDeleter>
ScopedObject<T, TDeleter> CreateScoped(TDeleter deleter) { return ScopedObject<T, TDeleter>(deleter); }

template<typename T, typename TDeleter>
ScopedObject<T, TDeleter> CreateScoped(T ptr, TDeleter deleter) { return ScopedObject<T, TDeleter>(ptr, deleter); }

#define CREATE_SCOPED(_type, _func) Tempest::CreateScoped<_type>([](_type t) { _func(t); })
}

#endif /* PATTERNS_HH_ */
