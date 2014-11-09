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

#ifndef _TEMPEST_AST_HH_
#define _TEMPEST_AST_HH_

#include "tempest/utils/types.hh"
#include "tempest/utils/assert.hh"
#include "tempest/utils/memory.hh"
#include "tempest/parser/driver-base.hh"

#include <type_traits>
#include <vector>

using std::is_const;

namespace Tempest
{
namespace AST
{
enum ASTNodeType
{
    TGE_AST_UNKNOWN=0,
    TGE_AST_LIST_ELEMENT,
    TGE_AST_BLOCK,
    TGE_AST_IDENTIFIER,
	TGE_AST_BOOLEAN,
    TGE_AST_INTEGER,
    TGE_AST_UNSIGNED,
    TGE_AST_FLOAT,
    TGE_AST_STRING_LITERAL,
    TGE_AST_NODES,
    TGE_AST_REFERENCE = 1 << 31
};

class VisitorInterface;
class ListElement;
class Block;
template<class T> class Value;
class StringLiteral;
template<class T> class Reference;

typedef ListElement List;

template<class T> struct ASTNodeInfo;
#define TGE_AST_NODE_INFO(_node, node_idx, vtype) \
    template<> struct ASTNodeInfo<_node> { \
        static const size_t node_enum = node_idx; \
        typedef vtype visitor_type; \
    };
TGE_AST_NODE_INFO(ListElement, TGE_AST_LIST_ELEMENT, AST::VisitorInterface)
TGE_AST_NODE_INFO(Block, TGE_AST_BLOCK, AST::VisitorInterface)
TGE_AST_NODE_INFO(Value<string>, TGE_AST_IDENTIFIER, AST::VisitorInterface)
TGE_AST_NODE_INFO(Value<bool>, TGE_AST_BOOLEAN, AST::VisitorInterface)
TGE_AST_NODE_INFO(Value<int>, TGE_AST_INTEGER, AST::VisitorInterface)
TGE_AST_NODE_INFO(Value<unsigned>, TGE_AST_UNSIGNED, AST::VisitorInterface)
TGE_AST_NODE_INFO(Value<float>, TGE_AST_FLOAT, AST::VisitorInterface)
TGE_AST_NODE_INFO(StringLiteral, TGE_AST_STRING_LITERAL, AST::VisitorInterface)

template<class T>
struct ASTNodeInfo<Reference<T>>
{
    static const size_t node_enum = ASTNodeInfo<T>::node_enum | TGE_AST_REFERENCE;
};

enum PrintFlags
{
	TGE_AST_PRINT_LINE_LOCATION = 1 << 0
};

// NOTES:
// Here the whole extra layer of abstraction is to reason easier about memory management.
// You might curse me because I could have used arena allocator instead. However, the
// parser generates some garbage nodes which I am not interested in managing in a
// separate area at this particular point in time. When I start optimizing I might
// consider it. Don't forget that the whole thing is just a part of intermediate
// application that is part of the pipeline. (cefx_compiler)


/*! \brief This kind of nodes are used for types that are managed by the object pool.
 * 
 *  Most nodes don't need to be specified as such, but in some cases such as variables,
 *  types and other similar stuff there is not unique representation of the specified node.
 *  In that case a reference is used.
 * 
 *  \remarks It is not intended to be used as member type. Use pointer instead and don't
 *           additionally complicate the system.
 *  \tparam T the type of the node that this object is going to reference.
 */
template<class T>
class Reference
{
    T* m_Data;
public:
    explicit Reference(T* ptr)
        :   m_Data(ptr) {}
    
    bool isBlockStatement() const { return m_Data->isBlockStatement(); }
    string getNodeName() const { return m_Data->getNodeName(); }
    
    T* get() { return m_Data; }
    const T* get() const { return m_Data; }
};


class NodeImpl
{
	Location	m_Location;
public:
    NodeImpl(Location loc)
        :   m_Location(loc) {}
    virtual ~NodeImpl() {}
    
    virtual size_t getNodeType() const=0;
    virtual bool isBlockStatement() const=0;
    virtual string getNodeName() const=0;

    virtual void accept(VisitorInterface* visitor) const=0;

    /*! \brief Get the AST::Location at which this node was declared.
        * 
        *  \note Built-in nodes don't have a valid AST::Location. You may check whether it
        *        is a built-in node by calling isBuiltIn.
        */
    Location getDeclarationLocation() const { return m_Location; }
private:
    NodeImpl(const NodeImpl&);
    NodeImpl& operator=(const NodeImpl&);
};

template<class T>
struct NodeImplModel: public NodeImpl
{
    T           m_Data;

    template<class... TArgs>
    NodeImplModel(Location loc, TArgs&&... args)
        :   NodeImpl(loc),
            m_Data(std::forward<TArgs>(args)...) {}

    virtual ~NodeImplModel() {}

    virtual size_t getNodeType() const override { return ASTNodeInfo<T>::node_enum; }
    virtual bool isBlockStatement() const override { return m_Data.isBlockStatement(); }
    virtual string getNodeName() const override { return m_Data.getNodeName(); }

    virtual void accept(VisitorInterface* visitor) const override { static_cast<typename ASTNodeInfo<T>::visitor_type*>(visitor)->visit(&m_Data); }
private:
    NodeImplModel(const NodeImplModel&);
    NodeImplModel& operator=(const NodeImplModel&);
};

template<class T>
struct NodeImplModel<Reference<T>>: public NodeImpl
{
    Reference<T> m_Data;

    template<class... TArgs>
    NodeImplModel(Location loc, TArgs&&... args)
        :   NodeImpl(loc),
            m_Data(std::forward<TArgs>(args)...) {}

    virtual ~NodeImplModel() {}

    virtual size_t getNodeType() const override { return ASTNodeInfo<Reference<T>>::node_enum; }
    virtual bool isBlockStatement() const override { return m_Data.isBlockStatement(); }
    virtual string getNodeName() const override { return m_Data.getNodeName(); }
    
    virtual void accept(VisitorInterface* visitor) const override { static_cast<typename ASTNodeInfo<T>::visitor_type*>(visitor)->visit(m_Data.get()); }
private:
    NodeImplModel(const NodeImplModel&);
    NodeImplModel& operator=(const NodeImplModel&);
};

template<class TValue>
struct NodeImplModelSelector
{
    typedef NodeImplModel<TValue> node_model;
    typedef TValue*               value_type;
    
    static value_type ExtractData(node_model* _model)
    {
        return _model ? &_model->m_Data : nullptr;
    }

    static const value_type ExtractData(const node_model* _model)
    {
        return _model ? &_model->m_Data : nullptr;
    }
};

template<class TValue>
struct NodeImplModelSelector<Reference<TValue>>
{
    typedef NodeImplModel<Reference<TValue>> node_model;
    typedef TValue*                          value_type;    
        
    static value_type ExtractData(node_model* _model)
    {
        return _model ? _model->m_Data.get() : nullptr;
    }

    static const value_type ExtractData(const node_model* _model)
    {
        return _model ? _model->m_Data.get() : nullptr;
    }
};

template<>
struct NodeImplModelSelector<void>
{
	typedef NodeImpl               node_model;
    typedef void*                  value_type;
    
    static void ExtractData(const NodeImpl*) {}
};

#define TGE_DEBUG_AST_NODE

template<class T> struct NodeT;

typedef void (*NodeDeleter)(void* ptr);

void GenericDeleter(void* ptr);

//! \brief Raw pointers with some code sugar to make interfacing with the preferred way of memory management.
struct UnmanagedNode
{
    NodeImpl*	m_Impl;
    NodeDeleter m_Deleter; //! Used for deleting the content carried by this node and also for marking node as incomplete, aka already released or never being managed.
    
    UnmanagedNode()
        :   m_Impl(nullptr),
            m_Deleter(nullptr) {}
    
    UnmanagedNode(NodeImpl* ptr, NodeDeleter _deleter)
        :   m_Impl(ptr),
            m_Deleter(_deleter != nullptr ? _deleter : &GenericDeleter) {}
        
    UnmanagedNode(const UnmanagedNode& _node)
        :   m_Impl(_node.m_Impl),
            m_Deleter(_node.m_Deleter) {}

    UnmanagedNode& operator=(const UnmanagedNode& _node)
    {
        m_Impl = _node.m_Impl;
        m_Deleter = _node.m_Deleter;
        return *this;
    }

    void destroy()
    {
        if(m_Deleter)
            m_Deleter(m_Impl);
        else
            TGE_ASSERT(m_Impl == nullptr, "Possible memory leak.");
        m_Impl = nullptr;
        m_Deleter = nullptr;
    }
    
    void release()
    {
        m_Impl = nullptr;
        m_Deleter = nullptr;
    }

    /*! \remarks This was introduced as some way to fix the Bison parser without resorting to some assumptions
     *           which would inevitably get forgotten
     */
    bool isIncomplete()
    {
        TGE_ASSERT((m_Impl == nullptr && (m_Deleter == nullptr || m_Deleter == GenericDeleter)) || (m_Impl != nullptr && m_Deleter != &GenericDeleter), "Unexpected combination ");
        return m_Deleter == nullptr;
    }

    template<class TOther>
    UnmanagedNode(NodeT<TOther>&& _node);
    
     template<class TOther>
    UnmanagedNode& operator=(NodeT<TOther>&& _node);

    // It is an error to move unmanaged nodes. They are supposed to be copied and
    // manually managed. The main reason why they were created was the Bison parser
    // and its current implementation. If in the future I come up with better
    // parser in mind I am going to replace it.

#ifdef _MSC_VER
    // Here they are defined because containers insist to move stuff around.
    UnmanagedNode(UnmanagedNode&& _node)
        :   m_Impl(_node.m_Impl),
            m_Deleter(_node.m_Deleter) {}

    UnmanagedNode& operator=(UnmanagedNode&& _node)
    {
        m_Impl = _node.m_Impl;
        m_Deleter = _node.m_Deleter;
        return *this;
    }
#else
    UnmanagedNode(UnmanagedNode&& _node)=delete;
    UnmanagedNode& operator=(UnmanagedNode&& _node)=delete;
#endif
};

template<class T>
struct NodeT
{
    typedef NodeImplModelSelector<T>                node_model_details;
    typedef typename node_model_details::node_model node_model;
    typedef typename node_model_details::value_type value_type;
    node_model* m_Impl;
    NodeDeleter m_Deleter;

#ifdef TGE_DEBUG_AST_NODE
    bool        m_Robbed;
#endif

    NodeT()
        :   m_Impl(nullptr),
            m_Deleter(nullptr)
#ifdef TGE_DEBUG_AST_NODE
          , m_Robbed(false)
#endif
        {}

    NodeT(typename node_model_details::node_model* val, NodeDeleter _deleter)
        :   m_Impl(val),
            m_Deleter(_deleter)
#ifdef TGE_DEBUG_AST_NODE
          , m_Robbed(false)
#endif
        {}

    NodeT(UnmanagedNode&& _node)
        :   m_Impl(static_cast<typename node_model_details::node_model*>(_node.m_Impl)),
            m_Deleter(_node.m_Deleter == &GenericDeleter ? nullptr : _node.m_Deleter)
#ifdef TGE_DEBUG_AST_NODE
          , m_Robbed(false)
#endif
    {
        _node.m_Impl = nullptr;
        _node.m_Deleter = nullptr;
    }

    NodeT(NodeT&& _node)
        :   m_Impl(_node.m_Impl),
            m_Deleter(_node.m_Deleter)
#ifdef TGE_DEBUG_AST_NODE
          , m_Robbed(_node.m_Robbed)
#endif
    {
        TGE_ASSERT(this != &_node, "Misused move constructor");
        _node.m_Impl = nullptr;
        _node.m_Deleter = nullptr;
#ifdef TGE_DEBUG_AST_NODE
        TGE_ASSERT(!m_Robbed, "Moving in node with content that were already moved out");
        _node.m_Robbed = true;
#endif
    }
    
    template<class TOther>
    NodeT(NodeT<TOther>&& _node)
        :   m_Impl(static_cast<typename node_model_details::node_model*>(_node.m_Impl)),
            m_Deleter(_node.m_Deleter)
#ifdef TGE_DEBUG_AST_NODE
          , m_Robbed(_node.m_Robbed)
#endif
    {
        _node.m_Impl = nullptr;
        _node.m_Deleter = nullptr;
#ifdef TGE_DEBUG_AST_NODE
        TGE_ASSERT(!m_Robbed, "Moving in node with content that were already moved out");
        _node.m_Robbed = true;
#endif        
    }

    ~NodeT()
    {
        if(m_Deleter)
            m_Deleter(m_Impl);
        TGE_ASSERT(((m_Deleter == nullptr) == (m_Impl == nullptr)), "Invalid pointer or deleter. Either way you are leaking memory.");
    }

    NodeT& operator=(NodeT&& _node)
    {
        TGE_ASSERT(this != &_node, "Misused move constructor");
        if(m_Deleter)
            m_Deleter(m_Impl);
        m_Impl = static_cast<typename node_model_details::node_model*>(_node.m_Impl);
        m_Deleter = _node.m_Deleter;
        _node.m_Impl = nullptr;
        _node.m_Deleter = nullptr;
#ifdef TGE_DEBUG_AST_NODE
        m_Robbed = _node.m_Robbed;
        TGE_ASSERT(!m_Robbed, "Moving in node with content that were already moved out");
        _node.m_Robbed = true;
#endif
        return *this;
    }

    template<class TOther>
    NodeT& operator=(NodeT<TOther>&& _node)
    {
        if(m_Deleter)
            m_Deleter(m_Impl);
        m_Impl = static_cast<typename node_model_details::node_model*>(_node.m_Impl);
        m_Deleter = _node.m_Deleter;
        _node.m_Impl = nullptr;
        _node.m_Deleter = nullptr;
#ifdef TGE_DEBUG_AST_NODE
        m_Robbed = _node.m_Robbed;
        TGE_ASSERT(!m_Robbed, "Moving in node with content that were already moved out");
        _node.m_Robbed = true;
#endif
        return *this;
    }

    void destroy()
    {
#ifdef TGE_DEBUG_AST_NODE
//        TGE_ASSERT(!m_Impl, "Destroying legit content");
#endif
        if(m_Deleter)
            m_Deleter(m_Impl);
        m_Impl = nullptr;
        m_Deleter = nullptr;
    }

    size_t getNodeType() const { return m_Impl->getNodeType() & ~TGE_AST_REFERENCE; }
    bool isBlockStatement() const { return m_Impl->isBlockStatement(); }

    string getNodeName() const { return m_Impl->getNodeName(); }

    /*! \brief Get the AST::Location at which this node was declared.
     * 
     *  \note Built-in nodes don't have a valid AST::Location. You may check whether it
     *        is a built-in node by calling isBuiltIn.
     */
    Location getDeclarationLocation() const { return m_Impl->getDeclarationLocation(); }

    value_type get()
    {
        TGE_ASSERT(m_Impl == nullptr || m_Impl->getNodeType() == ASTNodeInfo<T>::node_enum, "You've made a mistake. Probably cast to wrong node type.");
#ifdef TGE_DEBUG_AST_NODE
        TGE_ASSERT(!m_Robbed, "The content of this node were moved");
#endif
        return node_model_details::ExtractData(m_Impl);
    } 
    const value_type get() const
    {
#ifdef TGE_DEBUG_AST_NODE
        TGE_ASSERT(!m_Robbed, "The content of this node were moved");
#endif
        return node_model_details::ExtractData(m_Impl);
    }

    value_type operator->()
    {
#ifdef TGE_DEBUG_AST_NODE
        TGE_ASSERT(!m_Robbed, "The content of this node were moved");
#endif
        return node_model_details::ExtractData(m_Impl);
    }
    const value_type operator->() const
    {
#ifdef TGE_DEBUG_AST_NODE
        TGE_ASSERT(!m_Robbed, "The content of this node were moved");
#endif
        return node_model_details::ExtractData(m_Impl);
    }

    template<class U>
    U* extract()
    {
#ifdef TGE_DEBUG_AST_NODE
        TGE_ASSERT(!m_Robbed, "The content of this node were moved");
#endif
        if(!m_Impl)
            return nullptr;
        auto node_type = m_Impl->getNodeType();
        TGE_ASSERT((node_type & ~TGE_AST_REFERENCE) == ASTNodeInfo<U>::node_enum, "Unexpected node");
        return (node_type & TGE_AST_REFERENCE) != 0 ?
                   static_cast<NodeImplModel<Reference<U>>*>(m_Impl)->m_Data.get() :
                   &static_cast<NodeImplModel<U>*>(m_Impl)->m_Data;
    }

    template<class U>
    const U* extract() const
    {
#ifdef TGE_DEBUG_AST_NODE
        TGE_ASSERT(!m_Robbed, "The content of this node were moved");
#endif
        if(!m_Impl)
            return nullptr;
        auto node_type = m_Impl->getNodeType();
        TGE_ASSERT((node_type & ~TGE_AST_REFERENCE) == ASTNodeInfo<U>::node_enum, "Unexpected node");
        return (node_type & TGE_AST_REFERENCE) != 0 ?
                   static_cast<NodeImplModel<Reference<U>>*>(m_Impl)->m_Data.get() :
                   &static_cast<NodeImplModel<U>*>(m_Impl)->m_Data;
    }

    void accept(VisitorInterface* visitor) const { m_Impl->accept(visitor); }

    operator bool() const
    {
        return m_Impl != nullptr;
    }

    bool operator!() const
    {
        return m_Impl == nullptr;
    }

    NodeT(const NodeT&) = delete;
    NodeT& operator=(const NodeT&) = delete;
};

typedef NodeT<void> Node;

template<class TOther>
UnmanagedNode::UnmanagedNode(NodeT<TOther>&& _node)
    :   m_Impl(_node.m_Impl),
        m_Deleter(_node.m_Deleter != nullptr ? _node.m_Deleter : &GenericDeleter)
{
    _node.m_Impl = nullptr;
    _node.m_Deleter = nullptr;
}

template<class TOther>
UnmanagedNode& UnmanagedNode::operator=(NodeT<TOther>&& _node)
{
    m_Impl = _node.m_Impl;
    m_Deleter = _node.m_Deleter != nullptr ? _node.m_Deleter : &GenericDeleter;
    _node.m_Impl = nullptr;
    _node.m_Deleter = nullptr;
    return *this;
}

template<class TNode, class... TArgs>
NodeT<TNode> CreateNodeTyped(Location loc, TArgs&&... args)
{
    return NodeT<TNode>(new NodeImplModel<TNode>(loc, std::forward<TArgs>(args)...), reinterpret_cast<NodeDeleter>(&DeallocFunction<NodeImplModel<TNode>>));
}

template<class TNode, class... TArgs>
Node CreateNode(Location loc, TArgs&&... args)
{
    return Node(new NodeImplModel<TNode>(loc, std::forward<TArgs>(args)...), reinterpret_cast<NodeDeleter>(&DeallocFunction<NodeImplModel<TNode>>));
}

template<class T>
class Value
{
    T                       m_Value;
public:
    Value(T t)
        :   m_Value(t) {}
     ~Value() {}

    string getNodeName() const { return "<literal>"; }
     
    T getValue() const { return m_Value; }
    bool isBlockStatement() const { return false; }
private:
    Value(const Value&);
    Value& operator=(const Value&);
};

class StringLiteral
{
    string                  m_Value;
public:
    StringLiteral(string str);
     ~StringLiteral();

    string getNodeName() const { return "<string>"; }
        
    string getValue() const { return m_Value; }
        
    bool isBlockStatement() const;
private:
    StringLiteral(const StringLiteral&);
    StringLiteral& operator=(const StringLiteral&);
};

class VisitorInterface
{
public:
    VisitorInterface() {}
    virtual ~VisitorInterface() {}
    
    virtual void visit(const Location& loc)=0;
    virtual void visit(const Value<float>* value)=0;
    virtual void visit(const Value<int>* value)=0;
    virtual void visit(const Value<unsigned>* value)=0;
    virtual void visit(const Value<bool>* value)=0;
    virtual void visit(const Value<string>* value)=0;
    virtual void visit(const ListElement* lst)=0;
    virtual void visit(const Block* _block)=0;
    virtual void visit(const StringLiteral* value)=0;
};

class PrinterInfrastructure
{
public:
    class IndentationRemover
    {
        size_t& m_Indentation;
    public:
        IndentationRemover(size_t& indentation) : m_Indentation(indentation) {}
        void operator()() { TGE_ASSERT(m_Indentation, "Removing indentation from place where there is none"); --m_Indentation;}
    };
    
private:
    size_t          m_Indentation;
    std::ostream&   m_OutputStream;
    size_t          m_Flags;
public:
    PrinterInfrastructure(std::ostream& os, uint32 flags);
     ~PrinterInfrastructure();
    
    std::ostream& stream() { return m_OutputStream; }
    
    AtScopeExit<IndentationRemover> createScopedIndentation() { ++m_Indentation; return AtScopeExit<IndentationRemover>(m_Indentation); }

    void setIndentation(size_t indentation) { m_Indentation = indentation; }
    size_t getIndentation() const { return m_Indentation; }
    
    bool hasFlags(uint32 flags) { return (m_Flags & flags) != 0; }
};

// When you want to build your own printer. Just pick what functions out of these you are going to use and
// wrap them behind a visitor. You are going to need printing infrastructure also for convenience.
template<class T>
void PrintNode(PrinterInfrastructure* printer, const Value<T>* value) { printer->stream() << value->getValue(); }
void PrintNode(AST::VisitorInterface* visitor, PrinterInfrastructure* printer, const ListElement* lst);
void PrintNode(AST::VisitorInterface* visitor, PrinterInfrastructure* printer, const Block* _block);
void PrintNode(PrinterInfrastructure* printer, const StringLiteral* value);

inline void PrintLocation(PrinterInfrastructure* printer, const Location& loc, const char* filename = nullptr)
{
    // GLSL is stupid; we should parse numbers and do the look-up ourselves.
    auto& os = printer->stream();
    os << "#line " << loc.startLine; 
    if(filename)
        os << " \"" << filename << "\"\n";
    else
        os << " 0\n";
}

template<class T> class ListIterator;

enum ListType
{
    TGE_AST_COMMA_SEPARATED_LIST,
    TGE_AST_SEMICOLON_SEPARATED_LIST
};

class ListElement
{
    AST::Node                  m_Current;
    AST::NodeT<ListElement>    m_Next;
    ListType                   m_Type;
public:
    typedef ListIterator<AST::Node>       iterator;
    typedef ListIterator<const AST::Node> const_iterator;

    ListElement(ListType lt, AST::Node node, AST::NodeT<ListElement> next=AST::NodeT<ListElement>());
    ~ListElement();

    void set(ListType _type, AST::Node _node)
    {
        m_Type = _type;
        m_Current = std::move(_node);
    }

    ListType getFormat() const;

    AST::NodeT<ListElement>* next();
    const AST::NodeT<ListElement>* next() const;

    void erase_next();

    iterator current();
    iterator end();

    const_iterator current() const;
    const_iterator end() const;

    AST::Node* current_front();
    const AST::Node* current_front() const;
    AST::Node* back();
    const AST::Node* back() const;

    void push_front(AST::Node&& ptr);
    void push_back(AST::Node&& ptr);

    string getNodeName() const { return "<list>"; }

    bool isBlockStatement() const;
private:
    ListElement(const ListElement&);
    ListElement& operator=(const ListElement&);
};

typedef ListElement List;

template<bool TCond, class TTrue, class TFalse>
struct TemplateIf
{
    typedef TTrue result_type;
};

template<class TTrue, class TFalse>
struct TemplateIf<false, TTrue, TFalse>
{
    typedef TFalse result_type;
};

template<class T>
class ListIterator
{
public:
    typedef typename TemplateIf<is_const<T>::value,
                                const ListElement,
                                ListElement>::result_type node_type;
private:
    node_type* m_LElem;
public:
    typedef std::forward_iterator_tag       iterator_category;
    typedef T                               value_type;
    typedef T*                              pointer;
    typedef T&                              reference;
    typedef int                             difference_type;

    ListIterator()
        :   m_LElem(nullptr) {}

    ListIterator(node_type* lelem)
        :   m_LElem(lelem) {}

    ListIterator(const ListIterator& iter)
        :   m_LElem(iter.m_LElem) {}

    ~ListIterator() {}

    ListIterator& operator=(const ListIterator& iter)
    {
        m_LElem = iter.m_LElem;
        return *this;
    }

    bool operator==(const ListIterator& iter) const { return m_LElem == iter.m_LElem; }
    bool operator!=(const ListIterator& iter) const { return m_LElem != iter.m_LElem; }

    reference operator*() const { return static_cast<reference>(*(m_LElem->current_front())); }
    pointer operator->() const { return static_cast<pointer>(m_LElem->current_front()); }

    ListIterator& operator++()
    {
        if(m_LElem)
            m_LElem = m_LElem->next()->get();
        return *this;
    }

    ListIterator operator++(int)
    {
        ListIterator old_iter(*this);
        if(m_LElem)
            m_LElem = m_LElem->next()->get();
        return old_iter;
    }
    
    ListIterator operator+(difference_type diff)
    {
        ListIterator new_iterator(*this);
        new_iterator += diff;
        return new_iterator;
    }
    
    ListIterator& operator+=(difference_type diff)
    {
        TGE_ASSERT(diff > 0, "Forward iterator can't go backwards");
        for(difference_type i = 0; (i < diff) && m_LElem; ++i)
            m_LElem = m_LElem->next()->get();
        return *this;
    }
    
    operator bool() const { return m_LElem && m_LElem->current_front() != nullptr; }
    
    node_type* getElement() const { return m_LElem; }    
    pointer getNode() const { return m_LElem->current_front(); }
};

template<class T>
class NamedList
{
    string           m_Name;
    AST::NodeT<List> m_List;
public:
    NamedList(string name, AST::NodeT<List> _list)
        :   m_Name(name),
            m_List(std::move(_list)) {}
    ~NamedList() {}

    string getNodeName() const { return m_Name; }
    
    List* getBody() { return m_List.get(); }
    const List* getBody() const { return m_List.get(); }

    bool isBlockStatement() const { return true; }

    void printList(AST::VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const string& declaration) const
    {
        std::ostream& os = printer->stream();
        os << declaration << " " << m_Name << "\n";
        for(size_t i = 0, indentation = printer->getIndentation(); i < indentation; ++i)
            os << "\t";
        os << "{\n";
        if(m_List)
        {
            auto indent = printer->createScopedIndentation();
            visitor->visit(m_List.get());
        }
        for(size_t i = 0, indentation = printer->getIndentation(); i < indentation; ++i)
            os << "\t";
        os << "}\n";
    }
private:
    NamedList(const NamedList&);
    NamedList& operator=(const NamedList&);
};

class Block
{
    AST::NodeT<List>          m_Body;
public:
    Block(AST::NodeT<List> _body);
    ~Block();

    List* getBody();
    const List* getBody() const;

    string getNodeName() const { return "<block statement>"; }
    
     bool isBlockStatement() const;
private:
    Block(const Block&);
    Block& operator=(const Block&);
};

typedef std::vector<AST::Node> ObjectPoolType;
typedef std::vector<size_t>    StackType;

class Driver: public DriverBase
{
protected:
    typedef std::vector<size_t>    StackPointers;

    StackType               m_Stack;
    ObjectPoolType          m_ObjectPool;

    AST::Node               m_ASTRoot;

    size_t                  m_ErrorCount,
                            m_WarningCount;
public:
    Driver();
     ~Driver();

    template<class T>
    T* find(const string& name)
    {
        AST::Node* id = findIdentifier(name);
        return id->getNodeType() == ASTNodeInfo<T>::node_enum ?	id->extract<T>() : nullptr;
    }

    bool pushOnStack(AST::Node&& node);

    template<class T, class... TArgs>
    AST::NodeT<AST::Reference<T>> createStackNode(Location loc, TArgs&&... args)
    {
        auto _node = AST::CreateNode<T>(loc, std::forward<TArgs>(args)...);
        return pushOnStack(std::move(_node)) ?
            CreateNodeTyped<AST::Reference<T>>(loc, m_ObjectPool.back().extract<T>()) : AST::NodeT<AST::Reference<T>>();
    }
    
    void setASTRoot(AST::Node ast_root);
    AST::Node* getASTRoot();
    const AST::Node* getASTRoot() const;

    ///! \remarks Dangerous -- returns reference to object which is part of an array.
    AST::Node* findIdentifier(const string& name);
    
    string                  __FileName;
};
}
}

#endif /* _TEMPEST_AST_HH_ */
