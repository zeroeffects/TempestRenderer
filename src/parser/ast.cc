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

#include "tempest/parser/ast.hh"
#include "tempest/utils/memory.hh"
#include "tempest/utils/logging.hh"

namespace Tempest
{
namespace AST
{
void GenericDeleter(void* ptr) { TGE_ASSERT(ptr == nullptr, "Possible memory leak"); }

StringLiteral::StringLiteral(string str)
    :   m_Value(str) {}

StringLiteral::~StringLiteral() {}

bool StringLiteral::isBlockStatement() const
{
    return false;
}

Block::Block(AST::NodeT<List> _body)
    :   m_Body(std::move(_body)) {}

Block::~Block() {}

List* Block::getBody()
{
    return m_Body.get();
}

const List* Block::getBody() const
{
    return m_Body.get();
}

bool Block::isBlockStatement() const
{
    return true;
}

ListElement::ListElement(ListType lt, AST::Node node, NodeT<List> next)
    :   m_Current(std::move(node)),
        m_Next(std::move(next)),
        m_Type(lt)
{
    TGE_ASSERT(!m_Current || m_Current.getNodeType() != AST::TGE_AST_LIST_ELEMENT, "Don't build list of lists. That's too unspecific and bug prone."
                                                                                 "What if you actually intended to insert another node in the list"
                                                                                 "and instead you have set the current one incidentally.");
}

ListElement::~ListElement()
{
}

ListType ListElement::getFormat() const
{
	return m_Type;
}

NodeT<List>* ListElement::next()
{
    return &m_Next;
}

const NodeT<List>* ListElement::next() const
{
	return &m_Next;
}


void ListElement::erase_next()
{
	m_Next = std::move(m_Next->m_Next);
}

ListElement::iterator ListElement::current()
{
    return ListElement::iterator(this);
}

ListElement::iterator ListElement::end()
{
    return ListElement::iterator();
}

ListElement::const_iterator ListElement::current() const
{
    return ListElement::const_iterator(this);
}

ListElement::const_iterator ListElement::end() const
{
    return ListElement::const_iterator();
}

AST::Node* ListElement::current_front()
{
    return &m_Current;
}

const AST::Node* ListElement::current_front() const
{
    return &m_Current;
}

AST::Node* ListElement::back()
{
	return m_Next ? m_Next->back() : &m_Current;
}

const AST::Node* ListElement::back() const
{
	return m_Next ? m_Next->back() : &m_Current;
}

void ListElement::push_front(AST::Node&& _node)
{
    m_Next = AST::CreateNode<List>(m_Current.getDeclarationLocation(), m_Type, std::move(m_Current), std::move(m_Next));
    m_Current = std::move(_node);
}

void ListElement::push_back(AST::Node&& _node)
{
    if(m_Next)
        m_Next->push_back(std::move(_node));
    else
        m_Next = AST::CreateNode<List>(_node ? _node.getDeclarationLocation() : TGE_DEFAULT_LOCATION, m_Type, std::move(_node));
}

bool ListElement::isBlockStatement() const
{
    return false;
}

Driver::Driver()
{
}

Driver::~Driver()
{
}

void Driver::setASTRoot(AST::Node ast_root)
{
    m_ASTRoot = std::move(ast_root);
}

AST::Node* Driver::getASTRoot()
{
    return &m_ASTRoot;
}

const AST::Node* Driver::getASTRoot() const
{
    return &m_ASTRoot;
}

AST::Node* Driver::findIdentifier(const string& name)
{
    for(size_t i = 0; i < m_Stack.size(); ++i)
    {
        TGE_ASSERT(&m_ObjectPool[m_Stack[i]] != nullptr, "Empty stack object. Something is not right.");
		auto& obj = m_ObjectPool[m_Stack[i]];
		if(obj.getNodeName() == name)
            return &obj;
    }
    return nullptr;
}

bool Driver::pushOnStack(AST::Node&& node)
{
    string name = node.getNodeName();
    if(name.empty())
    {
        Log(LogLevel::Error, "Could not insert unknown object. This might be caused by previous error.");
        ++m_ErrorCount;
        return false;
        
    }
    for(size_t i = 0; i < m_Stack.size(); ++i)
    {
        TGE_ASSERT(&m_ObjectPool[m_Stack[i]] != nullptr, "Empty stack object. Something is not right.");
		if(m_ObjectPool[m_Stack[i]].getNodeName() == name)
        {
            Log(LogLevel::Error, "Could not push the following object: ", m_ObjectPool[m_Stack[i]].getNodeName(), "\n"
                                "\tIt was already inserted at this index: ", i);
            ++m_ErrorCount;
            return false;
        }
    }
	m_Stack.push_back(m_ObjectPool.size());
	m_ObjectPool.emplace_back(std::move(node));
    return true;
}


PrinterInfrastructure::PrinterInfrastructure(std::ostream& os, uint32 flags)
    :   m_Indentation(0),
        m_OutputStream(os),
        m_Flags(flags)
{
}

PrinterInfrastructure::~PrinterInfrastructure()
{
}

void PrintNode(PrinterInfrastructure* printer, const StringLiteral* ptr)
{
    printer->stream() << '"' << ptr->getValue() << '"';
}

void PrintNode(AST::VisitorInterface* visitor, PrinterInfrastructure* printer, const Block* ptr)
{
    printer->stream() << "{\n";
    {
        auto indent = printer->createScopedIndentation();
        visitor->visit(ptr->getBody());
    }
    for(size_t i = 0; i < printer->getIndentation(); ++i)
        printer->stream() << "\t";
    printer->stream() << "}\n";
}

void PrintNode(AST::VisitorInterface* visitor, PrinterInfrastructure* printer, const ListElement* ptr)
{
    auto& os = printer->stream();
    auto format = ptr->getFormat();
    for(auto i = ptr->current(), iend = ptr->end(); i != iend;)
    {
        if(format == ListType::SemicolonSeparated)
        {
            if(printer->hasFlags(TGE_AST_PRINT_LINE_LOCATION))
            {
                visitor->visit(i->getDeclarationLocation());
            }
            for(size_t i = 0; i < printer->getIndentation(); ++i)
                os << "\t";
        }

        TGE_ASSERT(*i, "Valid node expected. Bad parsing beforehand.");
        i->accept(visitor);

        if(format == ListType::CommaSeparated)
        {
            if(++i != ptr->end())
                os << ", ";
        }
        else
        {
            if(!i->isBlockStatement())
                os << ";\n";
            ++i;
        }
    }
}

}
}
