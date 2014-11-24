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

#include "tempest/shader/shader-ast.hh"
#include "tempest/shader/shader-driver.hh"

namespace Tempest
{
namespace Shader
{
Parentheses::Parentheses(AST::Node _node)
    :   m_Node(std::move(_node))
{
}

Parentheses::~Parentheses()
{
}

AST::Node* Parentheses::getExpression()
{
    return &m_Node;
}

const AST::Node* Parentheses::getExpression() const
{
    return &m_Node;
}

bool Parentheses::isBlockStatement() const
{
    return false;
}

FunctionCall::FunctionCall(const FunctionDeclaration* func, NodeT<List> arg_list)
    :   m_Function(func)
{
    if(arg_list)
    {
        auto* expr = arg_list->current_front()->extract<Expression>();
        m_Args = AST::CreateNodeTyped<List>(arg_list.getDeclarationLocation(), arg_list->getFormat(), std::move(expr->getSecond()), AST::NodeT<List>());
        List* last = m_Args.get();
        for(auto i = (*arg_list->next())->current(); i != arg_list->end(); ++i)
        {
            auto* expr = i->extract<Expression>();
            last->push_back(std::move(expr->getSecond()));
        }
    }
}

FunctionCall::~FunctionCall() {}

const FunctionDeclaration* FunctionCall::getFunction() const
{
    return m_Function;
}

List* FunctionCall::getArguments()
{
    return m_Args.get();
}

const List* FunctionCall::getArguments() const
{
    return m_Args.get();
}

bool FunctionCall::isBlockStatement() const
{
    return false;
}

FunctionDeclaration::FunctionDeclaration(const Type* return_type, string name, NodeT<List> var_list)
    :   m_ReturnType(return_type),
        m_Name(name),
        m_VarList(std::move(var_list)) {}

FunctionDeclaration::~FunctionDeclaration()
{
}

NodeT<FunctionCall> FunctionDeclaration::createFunctionCall(Location loc, NodeT<List> arg_list)
{
    List::iterator i, j;
    for(i = arg_list->current(), j = m_VarList->current();
        i != arg_list->end() && j != m_VarList->end(); ++i, ++j);
    return i == arg_list->end() && j == m_VarList->end() ? CreateNodeTyped<FunctionCall>(loc, this, std::move(arg_list)) : AST::NodeT<FunctionCall>();
}

bool FunctionDeclaration::sameParameters(const List* var_list) const
{
    List::const_iterator i;
    List::iterator j;
    for(i = var_list->current(), j = m_VarList->current();
        i != var_list->end() && j != m_VarList->end(); ++i, ++j)
    {
        if(!i || !j)
            return false;
        const AST::Node* decl = j->extract<Declaration>()->getVariables();
        auto* expr = i->extract<Expression>();
        if(expr == nullptr)
        {
            Tempest::Log(LogLevel::Error, "Request for function which contains invalid expression. Probable consequence of previous error.");
            return false;
        }
        if(!expr->getFirst()->hasBase(decl->extract<Variable>()->getType()))
            return false;
    }
    return i == var_list->end() && j == m_VarList->end();
}

const Type* FunctionDeclaration::getReturnType() const
{
    return m_ReturnType;
}

bool FunctionDeclaration::isBlockStatement() const
{
    return false;
}

Typedef::Typedef(Type* _type, string name)
    :   m_Name(name),
        m_Type(_type) {}

Typedef::~Typedef() {}

Type* Typedef::getType()
{
    return m_Type;
}

const Type* Typedef::getType() const
{
    return m_Type;
}

bool Typedef::isBlockStatement() const
{
    return false;
}

ConstructorCall::ConstructorCall(const Type* _type, NodeT<List> arg_list)
    :   m_Type(_type)
{
    if(arg_list)
    {
        auto* expr = arg_list->current_front()->extract<Expression>();
        m_Args = AST::CreateNodeTyped<List>(arg_list.getDeclarationLocation(), arg_list->getFormat(), std::move(expr->getSecond()), AST::NodeT<List>());
        auto* last = m_Args.get();
        for(auto i = (*arg_list->next())->current(); i != arg_list->end(); ++i)
        {
            auto* expr = i->extract<Expression>();
            last->push_back(std::move(expr->getSecond()));
        }
    }
}

ConstructorCall::~ConstructorCall() {}

const Type* ConstructorCall::getType() const
{
    return m_Type;
}

bool ConstructorCall::isBlockStatement() const
{
    return false;
}

ScalarType::ScalarType(bool integer, string name)
    :   m_Integer(integer),
        m_Name(name)
{
}

ScalarType::~ScalarType() {}

const Type* ScalarType::binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const
{
    switch(operandB->getTypeEnum())
    {
    case ElementType::Scalar:
    {
        switch(binop)
        {
        case TGE_EFFECT_ADD:
        case TGE_EFFECT_SUBTRACT:
        case TGE_EFFECT_MULTIPLY:
        case TGE_EFFECT_DIVIDE:
        {
            if(this_type->hasImplicitConversionTo(operandB))
                return operandB;
            else if(operandB->hasImplicitConversionTo(this_type))
                return this_type;
        } break;
        case TGE_EFFECT_ASSIGN:
        case TGE_EFFECT_ADD_ASSIGN:
        case TGE_EFFECT_SUB_ASSIGN:
        case TGE_EFFECT_DIV_ASSIGN:
        case TGE_EFFECT_MUL_ASSIGN:
        {
            if(operandB->hasImplicitConversionTo(this_type))
                return this_type;
        } break;
        case TGE_EFFECT_AND:
        case TGE_EFFECT_OR:
        case TGE_EFFECT_XOR:
        case TGE_EFFECT_LESS:
        case TGE_EFFECT_GREATER:
        case TGE_EFFECT_GEQUAL:
        case TGE_EFFECT_LEQUAL:
        case TGE_EFFECT_EQUAL:
        case TGE_EFFECT_NEQUAL:
        {
            if(operandB->hasImplicitConversionTo(this_type) || this_type->hasImplicitConversionTo(operandB))
                return driver.find("bool");
        } break;
        case TGE_EFFECT_COMMA:
            return operandB;
            break;
        default:
            break;
        }
    } break;
    case ElementType::Vector:
    {
        const VectorType* opBtype = operandB->extract<VectorType>();
        switch(binop)
        {
        case TGE_EFFECT_ADD:
        case TGE_EFFECT_SUBTRACT:
        case TGE_EFFECT_MULTIPLY:
        case TGE_EFFECT_DIVIDE:
        {
            if(this_type->hasImplicitConversionTo(operandB->extract<VectorType>()->getBasicType()))
                return operandB;
            else
            {
                std::stringstream ss;
                if(m_Name[0] == 'u' || m_Name[0] == 'i')
                    ss << m_Name[0];
                ss << "vec" << opBtype->getDimension();
                const Type* vec_type = driver.find(ss.str());
                TGE_ASSERT(vec_type != nullptr, "Unknown vector type.");
                if(operandB->hasImplicitConversionTo(vec_type))
                    return vec_type;
                else if(vec_type->hasImplicitConversionTo(operandB))
                    return operandB;
            }
        } break;
        case TGE_EFFECT_COMMA:
            return operandB;
            break;
        default:
            break;
        }
    } break;
    case ElementType::Matrix:
    {
        switch(binop)
        {
        case TGE_EFFECT_ADD:
        case TGE_EFFECT_SUBTRACT:
        case TGE_EFFECT_MULTIPLY:
        case TGE_EFFECT_DIVIDE:
            if(this_type->hasImplicitConversionTo(operandB->extract<MatrixType>()->getBasicType()))
                return operandB;
            break;
        case TGE_EFFECT_COMMA:
            return operandB;
            break;
        default:
            break;
        }
    } break;
    // ElementType::Array
    default:
        break;
    }
    
    if(!m_Integer)
        return nullptr;
    
    switch(operandB->getTypeEnum())
    {
    case ElementType::Scalar:
    {
        switch(binop)
        {
        case TGE_EFFECT_BITWISE_AND_ASSIGN:
        case TGE_EFFECT_BITWISE_XOR_ASSIGN:
        case TGE_EFFECT_BITWISE_OR_ASSIGN:
        case TGE_EFFECT_BITWISE_AND:
        case TGE_EFFECT_BITWISE_OR:
        case TGE_EFFECT_BITWISE_XOR:
        case TGE_EFFECT_MODULUS:
        {
            if(operandB == this_type)
                return operandB;
        } break;
        case TGE_EFFECT_BITWISE_SHIFT_RIGHT:
        case TGE_EFFECT_BITWISE_SHIFT_LEFT:
        {
            string basic_type = operandB->getNodeName();
            if(basic_type == "uint" || basic_type == "int")
                return this_type;
        } break;
        default:
            break;
        }
    } break;
    case ElementType::Vector:
    {
        const VectorType* opBtype = operandB->extract<VectorType>();
        switch(binop)
        {
        case TGE_EFFECT_BITWISE_AND:
        case TGE_EFFECT_BITWISE_OR:
        case TGE_EFFECT_BITWISE_XOR:
        case TGE_EFFECT_MODULUS:
        {
            const Type* basic_type = opBtype->getBasicType();
            if(basic_type == this_type)
                return basic_type;
        } break;
        default:
            break;
        }
    } break;
    default:
        break;
    }
    return nullptr;
}

const Type* ScalarType::unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const
{
    if(uniop == TGE_EFFECT_POSITIVE ||
       uniop == TGE_EFFECT_NEGATE ||
       uniop == TGE_EFFECT_PRE_INCR ||
       uniop == TGE_EFFECT_PRE_DECR ||
       uniop == TGE_EFFECT_POST_INCR ||
       uniop == TGE_EFFECT_POST_DECR ||
       uniop == TGE_EFFECT_NOT)
        return this_type;
    if(m_Integer && uniop == TGE_EFFECT_COMPLEMENT)
        return this_type;
    return nullptr;
}

bool ScalarType::hasValidConstructor(const List* var_list) const
{
    if(*var_list->next())
        return false;
    else
    {
		auto* expr = var_list->current_front()->extract<Expression>();
		ElementType type_enum = expr->getFirst()->getTypeEnum();
        if(type_enum != ElementType::Vector &&
           type_enum != ElementType::Scalar &&
           type_enum != ElementType::Matrix &&
           type_enum != ElementType::Array)
            return false;
    }
    return true;
}

bool ScalarType::hasImplicitConversionTo(const Type* _type) const
{
    return m_Integer && _type->getTypeEnum() == ElementType::Scalar && _type->getNodeName() == "float";
}

const Type* ScalarType::getMemberType(Driver& driver, const Type* this_type, const string& name) const { return nullptr; }
const Type* ScalarType::getArrayElementType() const { return nullptr; }

ArrayType::ArrayType(const Type* _type, AST::Node _size)
    :   m_Type(_type),
        m_Size(std::move(_size)) {}

ArrayType::~ArrayType() {}

AST::Node* ArrayType::getSize()
{
    return &m_Size;
}

const AST::Node* ArrayType::getSize() const
{
    return &m_Size;
}

const Type* ArrayType::getArrayElementType() const
{
    return m_Type;
}

bool ArrayType::hasValidConstructor(const List* var_list) const
{
    for(List::const_iterator i = var_list->current(), iend = var_list->end(); i != iend; ++i)
	{
		auto* expr = i->extract<Expression>();
		if(!expr->getFirst()->hasBase(m_Type))
            return false;
	}
    return true;
}

bool ArrayType::hasImplicitConversionTo(const Type* _type) const
{
    if(_type->getTypeEnum() != ElementType::Array)
        return false;
    return _type->extract<ArrayType>()->getArrayElementType() == m_Type;
}

const Type* ArrayType::binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const { return nullptr; }
const Type* ArrayType::unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const { return nullptr; }
const Type* ArrayType::getMemberType(Driver& driver, const Type* this_type, const string& name) const { return nullptr; }

VectorType::VectorType(const Type* _type, size_t vec_dim, string name)
    :   m_Name(name),
        m_Type(_type),
        m_VecDim(vec_dim)
{
}

VectorType::~VectorType() {}

size_t VectorType::getDimension() const
{
    return m_VecDim;
}

const Type* VectorType::getBasicType() const
{
    return m_Type;
}

const Type* VectorType::binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const
{
    switch(operandB->getTypeEnum())
    {
    case ElementType::Scalar:
    {
        switch(binop)
        {
        case TGE_EFFECT_ADD:
        case TGE_EFFECT_SUBTRACT:
        case TGE_EFFECT_MULTIPLY:
        case TGE_EFFECT_DIVIDE:
        {
            if(operandB->hasImplicitConversionTo(getBasicType()))
                return this_type;
            else
            {
                std::stringstream ss;
                if(m_Name[0] == 'u' || m_Name[0] == 'i')
                    ss << m_Name[0];
                ss << "vec" << this->getDimension();
                const Type* vec_type = driver.find(ss.str());
                TGE_ASSERT(vec_type != nullptr, "Unknown vector type.");
                if(this_type->hasImplicitConversionTo(vec_type))
                    return vec_type;
                else if(vec_type->hasImplicitConversionTo(this_type))
                    return this_type;
            }
        } break;
        case TGE_EFFECT_ADD_ASSIGN:
        case TGE_EFFECT_SUB_ASSIGN:
        case TGE_EFFECT_DIV_ASSIGN:
        case TGE_EFFECT_MUL_ASSIGN:
            if(operandB->hasImplicitConversionTo(getBasicType()))
                return this_type;
            break;
        case TGE_EFFECT_COMMA:
            return operandB;
        default:
            break;
        }
    } break;
    case ElementType::Vector:
    {
        switch(binop)
        {
        case TGE_EFFECT_MULTIPLY:
        case TGE_EFFECT_ADD:
        case TGE_EFFECT_SUBTRACT:
        case TGE_EFFECT_DIVIDE:
        {
            if(this_type->hasImplicitConversionTo(operandB))
                return operandB;
            else if(operandB->hasImplicitConversionTo(this_type))
                return this_type;
        } break;
        case TGE_EFFECT_ASSIGN:
        case TGE_EFFECT_ADD_ASSIGN:
        case TGE_EFFECT_SUB_ASSIGN:
        case TGE_EFFECT_DIV_ASSIGN:
        {
            if(operandB->hasImplicitConversionTo(this_type))
                return this_type;
        } break;
        case TGE_EFFECT_EQUAL:
        case TGE_EFFECT_NEQUAL:
            if(operandB->hasImplicitConversionTo(this_type) ||
               this_type->hasImplicitConversionTo(operandB))
                return driver.find("bool");
            break;
        case TGE_EFFECT_COMMA:
            return operandB;
            break;
        default:
            break;
        }
    } break;
    case ElementType::Matrix:
    {
        const MatrixType* opBtype = operandB->extract<MatrixType>();
        switch(binop)
        {
        case TGE_EFFECT_MULTIPLY:
            if(m_VecDim == opBtype->getRows())
            {
                std::stringstream ss;
                ss << "vec" << m_VecDim;
                return driver.find(ss.str());
            }
            break;
        case TGE_EFFECT_COMMA:
            return operandB;
            break;
        default:
            break;
        }
    } break;
    // ElementType::Array
    default:
        break;
    }
    
    if(!m_Type->extract<ScalarType>()->isInteger())
        return nullptr;
    
    switch(operandB->getTypeEnum())
    {
    case ElementType::Scalar:
    {
        switch(binop)
        {
        case TGE_EFFECT_BITWISE_AND_ASSIGN:
        case TGE_EFFECT_BITWISE_XOR_ASSIGN:
        case TGE_EFFECT_BITWISE_OR_ASSIGN:
        case TGE_EFFECT_BITWISE_AND:
        case TGE_EFFECT_BITWISE_OR:
        case TGE_EFFECT_BITWISE_XOR:
        case TGE_EFFECT_MODULUS:
        {
            if(operandB == m_Type)
                return this_type;
        } break;
        case TGE_EFFECT_BITWISE_SHIFT_RIGHT:
        case TGE_EFFECT_BITWISE_SHIFT_LEFT:
        {
            string basic_type = operandB->getNodeName();
            if(basic_type == "uint" || basic_type == "int")
                return this_type;
        } break;
        default:
            break;
        }
    } break;
    case ElementType::Vector:
    {
        const VectorType* opBtype = operandB->extract<VectorType>();
        switch(binop)
        {
        case TGE_EFFECT_BITWISE_AND_ASSIGN:
        case TGE_EFFECT_BITWISE_XOR_ASSIGN:
        case TGE_EFFECT_BITWISE_OR_ASSIGN:
        case TGE_EFFECT_BITWISE_AND:
        case TGE_EFFECT_BITWISE_OR:
        case TGE_EFFECT_BITWISE_XOR:
        case TGE_EFFECT_MODULUS:
        {
            if(operandB == this_type)
                return this_type;
        } break;
        case TGE_EFFECT_BITWISE_SHIFT_RIGHT:
        case TGE_EFFECT_BITWISE_SHIFT_LEFT:
        {
            string basic_type = opBtype->getBasicType()->getNodeName();
            if(opBtype->getDimension() == m_VecDim && (basic_type == "uint" || basic_type == "int"))
                return this_type;
        } break;
        default:
            break;
        }
    } break;
    default:
        break;
    }
    return nullptr;
}

const Type* VectorType::unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const
{
    if(uniop == TGE_EFFECT_POSITIVE ||
       uniop == TGE_EFFECT_NEGATE ||
       uniop == TGE_EFFECT_PRE_INCR ||
       uniop == TGE_EFFECT_PRE_DECR ||
       uniop == TGE_EFFECT_POST_INCR ||
       uniop == TGE_EFFECT_POST_DECR)
        return this_type;
    return m_Type->extract<ScalarType>()->isInteger() && uniop == TGE_EFFECT_COMPLEMENT ?
                this_type : nullptr;
}

const Type* VectorType::getMemberType(Driver& driver, const Type* this_type, const string& name) const
{
    static const char coord[3][4] = { { 'x', 'y', 'z', 'w' },
                                      { 'r', 'g', 'b', 'a' },
                                      { 's', 't', 'p', 'q' } }; // FIXME

    TGE_ASSERT(m_VecDim <= 4, "Unsupported vector dimension.");
    for(size_t i = 0; i < name.size(); ++i)
        for(size_t j = 0;; ++j)
        {
            if(j == m_VecDim)
                return nullptr;
            if(name[i] == coord[0][j] || name[i] == coord[1][j] || name[i] == coord[2][j])
                break;
        }
    if(name.size() == m_VecDim)
        return this_type;
    else if(name.size() == 1)
        return m_Type;

    string _prefix,
           basic_type = m_Type->getNodeName();

    if(basic_type[0] != 'f')
        _prefix = basic_type[0];
    std::stringstream ss;
    ss << _prefix << "vec" << name.size();
    return driver.find(ss.str());
}

const Type* VectorType::getArrayElementType() const
{
    return m_Type;
}

bool VectorType::hasValidConstructor(const List* var_list) const
{
    if(*var_list->next())
    {
        size_t total_element = 0;
        for(List::const_iterator i = var_list->current(), iend = var_list->end(); i != iend; ++i)
        {
			auto* expr = i->extract<Expression>();
            if(expr == nullptr)
            {
                Tempest::Log(LogLevel::Error, "Request for invalid constructor call. Probable consequence of previous error.");
                return false;
            }
			ElementType type_enum = expr->getFirst()->getTypeEnum();
            switch(type_enum)
            {
            case ElementType::Scalar: ++total_element; break;
            case ElementType::Vector: total_element += expr->getFirst()->extract<VectorType>()->getDimension(); break;
            case ElementType::Matrix:
            {
                const MatrixType* mat = expr->getFirst()->extract<MatrixType>();
                total_element += mat->getRows()*mat->getColumns();
            } break;
//          case ElementType::Array:
//          {
//              ArrayType& arr = static_cast<ArrayType&>(*i);
//              total_element // TODO: constants
//          } break;
            default:
                return false;
            }
        }
        if(m_VecDim != total_element)
            return false;
    }
    else
    {
        auto* expr = var_list->current_front()->extract<Expression>();
        if(expr == nullptr)
        {
            Tempest::Log(LogLevel::Error, "Request for invalid constructor call. Probable consequence of previous error.");
            return false;
        }
        ElementType type_enum = expr->getFirst()->getTypeEnum();
        if(type_enum != ElementType::Vector &&
           type_enum != ElementType::Scalar &&
           type_enum != ElementType::Matrix &&
           type_enum != ElementType::Array)
            return false;
    }
    return true;
}

bool VectorType::hasImplicitConversionTo(const Type* _type) const
{
    return m_Type->extract<ScalarType>()->isInteger() &&
           _type->getTypeEnum() == ElementType::Vector &&
           _type->extract<VectorType>()->getDimension() == m_VecDim &&
           _type->extract<VectorType>()->getBasicType()->getNodeName() == "float";
}

MatrixType::MatrixType(size_t rows, const Type* row_type, string name)
    :   m_Name(name),
        m_Rows(rows),
        m_RowType(row_type) {}

MatrixType::~MatrixType() {}

size_t MatrixType::getRows() const
{
    return m_Rows;
}

size_t MatrixType::getColumns() const
{
    return m_RowType->extract<VectorType>()->getDimension();
}

const Type* MatrixType::getBasicType() const
{
    return m_RowType->extract<VectorType>()->getBasicType();
}

const Type* MatrixType::binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const
{
    const Type* result_type = nullptr;
    switch(operandB->getTypeEnum())
    {
    case ElementType::Scalar:
    {
        switch(binop)
        {
        case TGE_EFFECT_ADD:
        case TGE_EFFECT_SUBTRACT:
        case TGE_EFFECT_MULTIPLY:
        case TGE_EFFECT_DIVIDE:
        case TGE_EFFECT_ADD_ASSIGN:
        case TGE_EFFECT_SUB_ASSIGN:
        case TGE_EFFECT_DIV_ASSIGN:
        case TGE_EFFECT_MUL_ASSIGN:
            if(operandB->hasImplicitConversionTo(getBasicType()))
                result_type = this_type;
            break;
        case TGE_EFFECT_COMMA:
            result_type = operandB;
            break;
        default:
            break;
        }
    } break;
    case ElementType::Vector:
    {
        const VectorType* opBtype = operandB->extract<VectorType>();
        switch(binop)
        {
        case TGE_EFFECT_MULTIPLY:
            if(getColumns() == opBtype->getDimension())
            {
                std::stringstream ss;
                ss << "vec" << opBtype->getDimension();
                result_type = driver.find(ss.str());
            }
            break;
        case TGE_EFFECT_COMMA:
            result_type = operandB;
            break;
        default:
            break;
        }
    } break;
    case ElementType::Matrix:
    {
        const MatrixType* opBtype = operandB->extract<MatrixType>();
        switch(binop)
        {
        case TGE_EFFECT_MULTIPLY:
        {
            if(getColumns() == opBtype->getRows())
            {
                std::stringstream ss;
                ss << "mat" << getRows() << "x" << opBtype->getColumns();
                result_type = driver.find(ss.str());
            }
        } break;
        case TGE_EFFECT_ADD:
        case TGE_EFFECT_SUBTRACT:
        case TGE_EFFECT_DIVIDE:
        case TGE_EFFECT_ASSIGN:
        case TGE_EFFECT_ADD_ASSIGN:
        case TGE_EFFECT_SUB_ASSIGN:
        case TGE_EFFECT_DIV_ASSIGN:
        {
            if(getColumns() == opBtype->getColumns() && getRows() == opBtype->getRows())
                result_type = this_type;
        } break;
        case TGE_EFFECT_EQUAL:
        case TGE_EFFECT_NEQUAL:
            if(getColumns() == opBtype->getColumns() && getRows() == opBtype->getRows())
                result_type = driver.find("bool");
            break;
        case TGE_EFFECT_COMMA:
            result_type = operandB;
            break;
        default:
            break;
        }
    } break;
    // ElementType::Array
    default:
        break;
    }
    return result_type;
}

const Type* MatrixType::unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const
{
    return uniop == TGE_EFFECT_POSITIVE ||
           uniop == TGE_EFFECT_NEGATE ||
           uniop == TGE_EFFECT_PRE_INCR ||
           uniop == TGE_EFFECT_PRE_DECR ||
           uniop == TGE_EFFECT_POST_INCR ||
           uniop == TGE_EFFECT_POST_DECR ?
                this_type : nullptr;
}

const Type* MatrixType::getArrayElementType() const
{
    return m_RowType;
}

bool MatrixType::hasValidConstructor(const List* var_list) const
{
    if(*var_list->next())
    {
        size_t max_element = getRows()*getColumns(), total_element = 0;
        for(List::const_iterator i = var_list->current(), iend = var_list->end(); i != iend; ++i)
        {
            auto* expr = i->extract<Expression>();
			ElementType type_enum = expr->getFirst()->getTypeEnum();
            switch(type_enum)
            {
            case ElementType::Scalar: ++total_element; break;
            case ElementType::Vector: total_element += expr->getFirst()->extract<VectorType>()->getDimension(); break;
            case ElementType::Matrix:
            {
                const MatrixType* mat = expr->getFirst()->extract<MatrixType>();
                total_element += mat->getRows()*mat->getColumns();
            } break;
//          case ElementType::Array:
//          {
//              ArrayType& arr = static_cast<ArrayType&>(*i);
//              total_element // TODO: constants
//          } break;
            default:
                return false;
            }
        }
        if(max_element != total_element)
            return false;
    }
    else
    {
		auto expr = var_list->current_front()->extract<Expression>();
		ElementType type_enum = expr->getFirst()->getTypeEnum();
        if(type_enum != ElementType::Vector &&
           type_enum != ElementType::Scalar &&
           type_enum != ElementType::Matrix &&
           type_enum != ElementType::Array)
            return false;
    }
    return true;
}

bool MatrixType::hasImplicitConversionTo(const Type* _type) const { return nullptr; }
const Type* MatrixType::getMemberType(Driver& driver, const Type* this_type, const string& name) const { return nullptr; } // TODO: there might be something out here

SamplerType::SamplerType(string name)
    :   m_Name(name) {}

SamplerType::~SamplerType() {}

bool SamplerType::hasValidConstructor(const List* var_list) const { return false; }
bool SamplerType::hasImplicitConversionTo(const Type* _type) const { return false; }
const Type* SamplerType::binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const { return nullptr; }
const Type* SamplerType::unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const { return nullptr; }
const Type* SamplerType::getMemberType(Driver& driver, const Type* this_type, const string& name) const { return nullptr; }
const Type* SamplerType::getArrayElementType() const { return nullptr; }

const Type* StructType::getMemberType(Driver& driver, const Type* this_type, const string& name) const
{
    for(auto iter = getBody()->current(), iter_end = getBody()->end(); iter != iter_end; ++iter)
    {
        TGE_ASSERT(iter->getNodeType() == TGE_EFFECT_DECLARATION, "Expecting declaration");
        auto* decl = iter->extract<Declaration>();
        
        auto* vars = decl->getVariables();
        switch(vars->getNodeType())
        {
        case TGE_EFFECT_VARIABLE:
        {
            auto* var = vars->extract<Variable>();
            if(var->getNodeName() == name)
            {
                return var->getType();
            }
        } break;
        case TGE_AST_LIST_ELEMENT:
        {
            auto* _list = vars->extract<List>();
            for(auto var_iter = _list->current(), var_iter_end = _list->end(); var_iter != var_iter_end; ++var_iter)
            {
                TGE_ASSERT(var_iter->getNodeType() == TGE_EFFECT_VARIABLE, "Expecting variable");
                auto* var = vars->extract<Variable>();
                if(var->getNodeName() == name)
                {
                    return var->getType();
                }
            }
        } break;
        default:
            TGE_ASSERT(false, "Unsupported node type"); break;
        }
    }
    return nullptr;
}

Variable::Variable(const Type* _type, string name)
    :   m_Name(name),
        m_Interpolation(InterpolationQualifier::Default),
        m_Storage(StorageQualifier::Default),
        m_Invariant(false),
		m_InvariantDecl(nullptr),
        m_Type(_type)
{
}

Variable::Variable(StorageQualifier _storage, const Type* _type, string name)
    :   m_Name(name),
        m_Interpolation(InterpolationQualifier::Default),
        m_Storage(_storage),
        m_Invariant(false),
		m_InvariantDecl(nullptr),
        m_Type(_type)
{
}

Variable::~Variable()
{
}

void Variable::setLayout(NodeT<List> _list)
{
    m_Layout = std::move(_list);
}

void Variable::setInterpolation(InterpolationQualifier ipl)
{
    m_Interpolation = ipl;
}

void Variable::setStorage(StorageQualifier storage)
{
    m_Storage = storage;
}

void Variable::setInvariant(bool val)
{
    m_Invariant = val;
}

List* Variable::getLayout()
{
    return m_Layout.get();
}

const List* Variable::getLayout() const
{
    return m_Layout.get();
}

InterpolationQualifier Variable::getInterpolation() const
{
    return m_Interpolation;
}

StorageQualifier Variable::getStorage() const
{
    return m_Storage;
}

bool Variable::getInvariant() const
{
    return m_Invariant;
}

const Type* Variable::getType() const
{
    return m_Type;
}

void Variable::setType(const Type* _type)
{
    m_Type = _type;
}

void Variable::setInvariantDeclaration(InvariantDeclaration* inv_dcl)
{
    m_InvariantDecl = inv_dcl;
}

bool Variable::isBlockStatement() const
{
    return false;
}

Declaration::Declaration(AST::Node var)
    :   m_Variables(std::move(var)) {}

Declaration::~Declaration() {}

AST::Node* Declaration::getVariables()
{
    return &m_Variables;
}

const AST::Node* Declaration::getVariables() const
{
    return &m_Variables;
}

void PrintType(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Variable* var)
{
    std::ostream& os = printer->stream();
    auto* layout = var->getLayout();
    if(layout)
    {
        os << "layout(";
        static_cast<AST::VisitorInterface*>(visitor)->visit(layout);
        os << ") ";
    }
    if(var->getInvariant())
        os << "invariant ";
    switch(var->getStorage())
    {
    case StorageQualifier::StructBuffer:
    case StorageQualifier::Default:
        break;
    case StorageQualifier::Const:
        os << "const "; break;
    case StorageQualifier::In:
        os << "in "; break;
    case StorageQualifier::CentroidIn:
        os << "centroid in "; break;
    case StorageQualifier::SampleIn:
        os << "sample in "; break;
    case StorageQualifier::Out:
        os << "out "; break;
    case StorageQualifier::CentroidOut:
        os << "centroid out "; break;
    case StorageQualifier::SampleOut:
        os << "sample out "; break;
    case StorageQualifier::InOut:
        os << "inout "; break;
    }
    switch(var->getInterpolation())
    {
    case InterpolationQualifier::Default:
        break;
    case InterpolationQualifier::Smooth:
        os << "smooth "; break;
    case InterpolationQualifier::Flat:
        os << "flat "; break;
    case InterpolationQualifier::Noperspective:
        os << "noperspective "; break;
    }
    visitor->visit(var->getType());
}

bool Declaration::isBlockStatement() const
{
    return false;
}

MemberVariable::MemberVariable(AST::Node parent, const Type* type, string name)
    :   m_Variable(type, name),
        m_Parent(std::move(parent)) {}

MemberVariable::~MemberVariable() {}

bool MemberVariable::isBlockStatement() const
{
    return false;
}

ArrayElementVariable::ArrayElementVariable(AST::Node parent, const Type* type, AST::Node expr)
    :   m_Variable(type, type->getNodeName() + "[]"),
        m_Parent(std::move(parent)),
        m_Expr(std::move(expr)) {}

ArrayElementVariable::~ArrayElementVariable() {}

bool ArrayElementVariable::isBlockStatement() const
{
    return false;
}

InvariantDeclaration::InvariantDeclaration(Variable* var)
    :   m_Variable(var)
{
    var->setInvariantDeclaration(this);
}

InvariantDeclaration::~InvariantDeclaration() {}

bool InvariantDeclaration::isBlockStatement() const
{
    return false;
}

FunctionDefinition::FunctionDefinition(const FunctionDeclaration* decl, AST::NodeT<List> body)
    :   m_Declaration(decl),
        m_Body(std::move(body))
{
}

FunctionDefinition::~FunctionDefinition()
{
}

bool FunctionDefinition::isBlockStatement() const
{
    return true;
}

FunctionSet::FunctionSet(string name)
    :   m_Name(name) {}

FunctionSet::~FunctionSet() {}

void FunctionSet::pushFunction(NodeT<FunctionDeclaration> func)
{
    m_Func.push_back(std::move(func));
}

FunctionDeclaration* FunctionSet::getFunction(const List* var_list)
{
    for(size_t i = 0; i < m_Func.size(); ++i)
        if(m_Func[i]->sameParameters(var_list))
            return m_Func[i].get();
    return nullptr;
}

FunctionDeclaration* FunctionSet::getFunction(size_t idx)
{
    return m_Func[idx].get();
}

const FunctionDeclaration* FunctionSet::getFunction(size_t idx) const
{
    return m_Func[idx].get();
}

size_t FunctionSet::getFunctionCount() const
{
    return m_Func.size();
}

bool FunctionSet::isBlockStatement() const
{
    return false;
}

BinaryOperator::BinaryOperator(BinaryOperatorType _type, AST::Node _first, AST::Node _second)
    :   m_Type(_type),
        m_First(std::move(_first)),
        m_Second(std::move(_second)) {}

BinaryOperator::~BinaryOperator() {}

BinaryOperatorType BinaryOperator::getOperation() const
{
    return m_Type;
}

bool BinaryOperator::isBlockStatement() const
{
    return false;
}

AST::Node* BinaryOperator::getLHSOperand()
{
    return &m_First;
}

const AST::Node* BinaryOperator::getLHSOperand() const
{
    return &m_First;
}

AST::Node* BinaryOperator::getRHSOperand()
{
    return &m_Second;
}

const AST::Node* BinaryOperator::getRHSOperand() const
{
    return &m_Second;
}

UnaryOperator::UnaryOperator(UnaryOperatorType _type, AST::Node _operand)
    :   m_Type(_type),
        m_Operand(std::move(_operand)) {}

UnaryOperator::~UnaryOperator() {}

bool UnaryOperator::isBlockStatement() const
{
    return false;
}

AST::Node* UnaryOperator::getOperand()
{
    return &m_Operand;
}

const AST::Node* UnaryOperator::getOperand() const
{
    return &m_Operand;
}

TernaryIf::TernaryIf(AST::Node cond, AST::Node true_expr, AST::Node false_expr)
    :   m_Condition(std::move(cond)),
        m_TrueExpr(std::move(true_expr)),
        m_FalseExpr(std::move(false_expr)) {}

TernaryIf::~TernaryIf() {}

bool TernaryIf::isBlockStatement() const
{
    return false;
}

IfStatement::IfStatement(AST::Node condition_statement, AST::Node true_statement, AST::Node false_statement)
    :   m_Condition(std::move(condition_statement)),
        m_TrueStatement(std::move(true_statement)),
        m_FalseStatement(std::move(false_statement)) {}

IfStatement::~IfStatement() {}

bool IfStatement::isBlockStatement() const
{
    return true;
}

WhileStatement::WhileStatement(AST::Node condition_statement, AST::Node statement, bool do_while)
    :   m_Condition(std::move(condition_statement)),
        m_Statement(std::move(statement)),
        m_IsDoWhile(do_while) {}

WhileStatement::~WhileStatement() {}

bool WhileStatement::isBlockStatement() const
{
    return true;
}

ForStatement::ForStatement(AST::Node init_statement, AST::Node condition_statement, AST::Node update_statement, AST::Node statement)
    :   m_Init(std::move(init_statement)),
        m_Condition(std::move(condition_statement)),
        m_Update(std::move(update_statement)),
        m_Statement(std::move(statement)) {}

ForStatement::~ForStatement() {}

bool ForStatement::isBlockStatement() const
{
    return true;
}

SwitchStatement::SwitchStatement(AST::Node condition_statement, NodeT<List> cases)
    :   m_Condition(std::move(condition_statement)),
        m_Cases(std::move(cases)) {}

SwitchStatement::~SwitchStatement() {}

bool SwitchStatement::isBlockStatement() const
{
    return true;
}

CaseStatement::CaseStatement(AST::Node label, NodeT<List> statement)
    :   m_Label(std::move(label)),
        m_Statement(std::move(statement)) {}

CaseStatement::~CaseStatement() {}

bool CaseStatement::isBlockStatement() const
{
    return false;
}

Technique::Technique(string name, NodeT<List> body)
    :   NamedList<Technique>(name, std::move(body)) {}

Technique::~Technique() {}

ShaderDeclaration::ShaderDeclaration(ShaderType _type, string name, NodeT<List> body)
    :   NamedList(name, std::move(body)),
        m_Type(_type) {}

ShaderDeclaration::~ShaderDeclaration() {}

ShaderType ShaderDeclaration::getType() const
{
    return m_Type;
}

bool ShaderDeclaration::hasBase(const Type* _type) const
{
    return _type->getNodeName() == "shader";
}

bool ShaderDeclaration::hasImplicitConversionTo(const Type* _type) const
{
    return (_type->getTypeEnum() == ElementType::Shader && _type->getNodeName() == "shader");
}

bool ShaderDeclaration::hasValidConstructor(const List* var_list) const
{
    return !var_list;
}

const Type* ShaderDeclaration::binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const { return nullptr; }
const Type* ShaderDeclaration::unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const { return nullptr; }
const Type* ShaderDeclaration::getMemberType(Driver& driver, const Type* this_type, const string& name) const { return nullptr; }
const Type* ShaderDeclaration::getArrayElementType() const { return nullptr; }

CompiledShader::CompiledShader() {}

CompiledShader::~CompiledShader() {}

bool CompiledShader::hasValidConstructor(const List* var_list) const { return false; }
bool CompiledShader::hasImplicitConversionTo(const Type* _type) const { return false; }
const Type* CompiledShader::binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const { return nullptr; }
const Type* CompiledShader::unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const { return nullptr; }
const Type* CompiledShader::getMemberType(Driver& driver, const Type* this_type, const string& name) const { return nullptr; }
const Type* CompiledShader::getArrayElementType() const { return nullptr; }

Profile::Profile() {}

Profile::~Profile() {}

bool Profile::hasValidConstructor(const List* var_list) const
{
	auto* expr = var_list->current_front()->extract<Expression>();
	if(!var_list || expr->getSecond().getNodeType() != TGE_AST_INTEGER)
        return false;

    for(List::const_iterator i = var_list->current(), iend = var_list->end(); i != iend; ++i)
	{
		auto* expr = i->extract<Expression>();
        if(expr->getSecond().getNodeType() != TGE_AST_IDENTIFIER)
            return false;
	}
    return true;
}

bool Profile::hasImplicitConversionTo(const Type* _type) const { return false; }

const Type* Profile::binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const { return nullptr; }
const Type* Profile::unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const { return nullptr; }
const Type* Profile::getMemberType(Driver& driver, const Type* this_type, const string& name) const { return nullptr; }
const Type* Profile::getArrayElementType() const { return nullptr; }

Pass::Pass(string name, NodeT<List> body)
    :   NamedList<Pass>(name, std::move(body)) {}

Pass::~Pass() {}

Import::Import(string name, NodeT<List> body)
    :   NamedList<Import>(name, std::move(body)) {}

Import::~Import() {}

Buffer::Buffer(string name, NodeT<List> body)
    :   NamedList<Buffer>(name, std::move(body)) {}

Buffer::~Buffer() {}

JumpStatement::JumpStatement(JumpStatementType jump_type)
    :   m_JumpType(jump_type)
{
}

JumpStatement::~JumpStatement()
{
}

bool JumpStatement::isBlockStatement() const
{
    return false;
}

ReturnStatement::ReturnStatement(AST::Node retexpr)
    :   m_ReturnExpression(std::move(retexpr))
{
}

ReturnStatement::~ReturnStatement()
{
}

Printer::Printer(std::ostream& os, uint32 flags)
    :   m_Printer(os, flags) {}
Printer::~Printer() {}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Typedef* _typedef)
{
    auto& os = printer->stream();
    os << "typedef ";
    visitor->visit(_typedef->getType());
    os << " " << _typedef->getNodeName();
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Parentheses* parentheses)
{
    auto& os = printer->stream();
    os << "(";
    parentheses->getExpression()->accept(visitor);
    os << ")";
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const FunctionDeclaration* func_decl)
{
    auto& os = printer->stream();
    auto* ret_type = func_decl->getReturnType();
    if(ret_type)
        visitor->visit(ret_type);
    else
        os << "void";
    os << " " << func_decl->getNodeName() << "(";
    for(auto i = func_decl->getParametersBegin(), ibegin = i, iend = func_decl->getParametersEnd(); i != iend; ++i)
    {
        if(i != ibegin)
            os << ", ";
        i->accept(visitor);
    }
    os << ")";
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const FunctionDefinition* func_def)
{
    auto& os = printer->stream();
    auto* func_decl = func_def->getDeclaration();
    visitor->visit(func_decl);
    os << "\n";
    size_t indentation = printer->getIndentation();
    for(size_t i = 0; i < indentation; ++i)
        os << "\t";
    os << "{\n";
    auto* body = func_def->getBody();
    if(body)
    {
        auto indent = printer->createScopedIndentation();
        for(size_t i = 0; i < indentation; ++i)
            os << "\t";
        static_cast<AST::VisitorInterface*>(visitor)->visit(body);
    }
    for(size_t i = 0; i < indentation; ++i)
        os << "\t";
    os << "}\n";
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const FunctionCall* func_call)
{
    auto& os = printer->stream();
    os << func_call->getNodeName() << "(";
    auto* args = func_call->getArguments();
    if(args)
        static_cast<AST::VisitorInterface*>(visitor)->visit(args);
    os << ")";
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ConstructorCall* constructor)
{
    auto& os = printer->stream();
    visitor->visit(constructor->getType());
    os << "(";
    auto* args = constructor->getArguments();
    if(args)
        static_cast<AST::VisitorInterface*>(visitor)->visit(args);
    os << ")";
}

void PrintNode(AST::PrinterInfrastructure* printer, const ScalarType* scalar_type)
{
    printer->stream() << scalar_type->getNodeName();
}

void PrintNode(AST::PrinterInfrastructure* printer, const VectorType* vector_type)
{
    printer->stream() << vector_type->getNodeName();
}

void PrintNode(AST::PrinterInfrastructure* printer, const MatrixType* matrix_type)
{
    printer->stream() << matrix_type->getNodeName();
}

void PrintNode(AST::PrinterInfrastructure* printer, const SamplerType* sampler_type)
{
    printer->stream() << sampler_type->getNodeName();
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ArrayType* array_type)
{
    visitor->visit(array_type->getBasicType());
}

void PrintNode(AST::PrinterInfrastructure* printer, const Variable* var)
{
    printer->stream() << var->getNodeName();
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Declaration* decl)
{
    // TODO: Array
    auto& os = printer->stream();
    auto* variables = decl->getVariables();
    auto var_type = variables->getNodeType();
    switch(var_type)
    {
    case TGE_EFFECT_VARIABLE:
    {
        auto* var = variables->extract<Variable>();
        PrintType(visitor, printer, var);
        if(!var->getNodeName().empty())
            os << " ";
        if(var->getType()->getTypeEnum() == ElementType::Array)
        {
            auto* array_type = var->getType()->extract<ArrayType>();
            visitor->visit(var);
            os << "[";
            auto* array_size = array_type->getSize();
            if(*array_size)
            {
                array_size->accept(visitor);
            }
            os << "]";
        }
        else
            visitor->visit(var);
    } break;
    case TGE_EFFECT_BINARY_OPERATOR:
    {
        auto* binop = variables->extract<BinaryOperator>();
        auto* var = binop->getLHSOperand()->extract<Variable>();
        PrintType(visitor, printer, var);
        os << " ";
        if(var->getType()->getTypeEnum() == ElementType::Array)
        {
            auto* array_type = var->getType()->extract<ArrayType>();
            visitor->visit(var);
            os << "[";
            array_type->getSize()->accept(visitor);
            os << "]=";
            binop->getRHSOperand()->accept(visitor);
        }
        else
            visitor->visit(binop);

    } break;
    case AST::TGE_AST_LIST_ELEMENT:
    {
        auto* _list = variables->extract<List>();
        auto* _node = _list->current_front();
        const Variable* var;
        if(_node->getNodeType() == TGE_EFFECT_VARIABLE)
        {
            var = _node->extract<Variable>();
        }
        else if(_node->getNodeType() == TGE_EFFECT_BINARY_OPERATOR)
        {
            var = _node->extract<BinaryOperator>()->getLHSOperand()->extract<Variable>();
        }
        else
            TGE_ASSERT(false, "unsupported");
        PrintType(visitor, printer, var);
        os << " ";
        for(auto i = _list->current();;)
        {
            if(i->getNodeType() == TGE_EFFECT_VARIABLE)
            {
                auto* ref_var = i->extract<Variable>();
                visitor->visit(ref_var);
                if(ref_var->getType()->getTypeEnum() == ElementType::Array)
                {
                    os << "[";
                    ref_var->getType()->extract<ArrayType>()->getSize()->accept(visitor);
                    os << "]";
                }
            }
            else if(i->getNodeType() == TGE_EFFECT_BINARY_OPERATOR)
            {
                const BinaryOperator* binop = i->extract<BinaryOperator>();
                var = binop->getLHSOperand()->extract<Variable>();
                visitor->visit(var);
                if(var->getType()->getTypeEnum() == ElementType::Array)
                {
                    os << "[";
                    var->getType()->extract<ArrayType>()->getSize()->accept(visitor);
                    os << "]=";
                }
                else
                    os << "=";
                binop->getRHSOperand()->accept(visitor);
            }
            else
                TGE_ASSERT(false, "unsupported");
            if(++i != _list->end())
                os << ", ";
            else
                break;
        }
    } break;
    case TGE_EFFECT_TYPE:
    {
        auto* _type = variables->extract<Type>();
        switch(_type->getTypeEnum())
        {
        case ElementType::Struct:
        {
            auto* _struct = _type->extract<StructType>();
            _struct->printList(visitor, printer, "struct");
        } break;
        }
    } break;
    default:
        TGE_ASSERT(false, "unsupported"); break;
    }
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const MemberVariable* mem_var)
{
    mem_var->getParent()->accept(visitor);
    printer->stream() << "." << mem_var->getVariable()->getNodeName();
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ArrayElementVariable* arr_var)
{
    auto& os = printer->stream();
    arr_var->getParent()->accept(visitor);
    os << "[";
    arr_var->getIndex()->accept(visitor);
    os << "]";
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const InvariantDeclaration* invar_decl)
{
    printer->stream() << "invariant ";
    visitor->visit(invar_decl->getVariable());
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const BinaryOperator* binop)
{
    string binary_operator;
    switch(binop->getOperation())
    {
    case TGE_EFFECT_ASSIGN:
        binary_operator = "="; break;
    case TGE_EFFECT_ADD_ASSIGN:
        binary_operator = "+="; break;
    case TGE_EFFECT_SUB_ASSIGN:
        binary_operator = "-="; break;
    case TGE_EFFECT_MUL_ASSIGN:
        binary_operator = "*="; break;
    case TGE_EFFECT_DIV_ASSIGN:
        binary_operator = "/="; break;
    case TGE_EFFECT_MOD_ASSIGN:
        binary_operator = "%="; break;
    case TGE_EFFECT_BITWISE_AND_ASSIGN:
        binary_operator = "&="; break;
    case TGE_EFFECT_BITWISE_XOR_ASSIGN:
        binary_operator = "^="; break;
    case TGE_EFFECT_BITWISE_OR_ASSIGN:
        binary_operator = "|="; break;
    case TGE_EFFECT_ADD:
        binary_operator = "+"; break;
    case TGE_EFFECT_SUBTRACT:
        binary_operator = "-"; break;
    case TGE_EFFECT_MULTIPLY:
        binary_operator = "*"; break;
    case TGE_EFFECT_DIVIDE:
        binary_operator = "/"; break;
    case TGE_EFFECT_MODULUS:
        binary_operator = "%"; break;
    case TGE_EFFECT_BITWISE_AND:
        binary_operator = "&"; break;
    case TGE_EFFECT_BITWISE_OR:
        binary_operator = "|"; break;
    case TGE_EFFECT_BITWISE_XOR:
        binary_operator = "^"; break;
    case TGE_EFFECT_BITWISE_SHIFT_RIGHT:
        binary_operator = ">>"; break;
    case TGE_EFFECT_BITWISE_SHIFT_LEFT:
        binary_operator = "<<"; break;
    case TGE_EFFECT_LESS:
        binary_operator = "<"; break;
    case TGE_EFFECT_GREATER:
        binary_operator = ">"; break;
    case TGE_EFFECT_LEQUAL:
        binary_operator = "<="; break;
    case TGE_EFFECT_GEQUAL:
        binary_operator = ">="; break;
    case TGE_EFFECT_OR:
        binary_operator = "||"; break;
    case TGE_EFFECT_AND:
        binary_operator = "&&"; break;
    case TGE_EFFECT_XOR:
        binary_operator = "^^"; break;
    case TGE_EFFECT_EQUAL:
        binary_operator = "=="; break;
    case TGE_EFFECT_NEQUAL:
        binary_operator = "!="; break;
    case TGE_EFFECT_COMMA:
        binary_operator = ","; break;
    }
    binop->getLHSOperand()->accept(visitor);
    printer->stream() << binary_operator;
    binop->getRHSOperand()->accept(visitor);   
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const UnaryOperator* unaryop)
{
    string unary_operator;
    bool is_post;
    switch(unaryop->getOperation())
    {
    case TGE_EFFECT_POSITIVE:
        unary_operator = "+"; is_post = false; break;
    case TGE_EFFECT_NEGATE:
        unary_operator = "-"; is_post = false; break;
    case TGE_EFFECT_NOT:
        unary_operator = "!"; is_post = false; break;
    case TGE_EFFECT_COMPLEMENT:
        unary_operator = "~"; is_post = false; break;
    case TGE_EFFECT_PRE_INCR:
        unary_operator = "++"; is_post = false; break;
    case TGE_EFFECT_PRE_DECR:
        unary_operator = "--"; is_post = false; break;
    case TGE_EFFECT_POST_INCR:
        unary_operator = "++"; is_post = true; break;
    case TGE_EFFECT_POST_DECR:
        unary_operator = "--"; is_post = true; break;
    }
    if(is_post)
    {
        unaryop->getOperand()->accept(visitor);
        printer->stream() << unary_operator;
    }
    else
    {
        printer->stream() << unary_operator;
        unaryop->getOperand()->accept(visitor);
    }
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const TernaryIf* ternary_if)
{
    auto& os = printer->stream();
    ternary_if->getCondition()->accept(visitor);
    os << " ? ";
    ternary_if->getTrueExpression()->accept(visitor);
    os << " : ";
    ternary_if->getFalseExpression()->accept(visitor);
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const WhileStatement* while_stmt)
{
    auto& os = printer->stream();
    auto indentation = printer->getIndentation();
    if(while_stmt->isDoWhile())
    {
        os << "do\n";
        for(size_t i = 0; i < indentation; ++i)
            os << "\t";
        if(!while_stmt->getLoopBody()->isBlockStatement())
            os << "\t";
        while_stmt->getLoopBody()->accept(visitor);
        for(size_t i = 0; i < indentation; ++i)
            os << "\t";
        os << "while(";
        while_stmt->getCondition()->accept(visitor);
        os << ");\n";
    }
    else
    {
        os << "while(";
        while_stmt->getCondition()->accept(visitor);
        os << ")\n";
        for(size_t i = 0; i < indentation; ++i)
            os << "\t";
        auto* loop_body = while_stmt->getLoopBody();
        if(!loop_body->isBlockStatement())
            os << "\t";
        loop_body->accept(visitor);
        if(!loop_body->isBlockStatement())
            os << ";\n";
    }
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ForStatement* for_stmt)
{
    auto& os = printer->stream();
    os << "for(";
    for_stmt->getInitStatement()->accept(visitor);
    os << "; ";
    for_stmt->getCondition()->accept(visitor);
    os << "; ";
    for_stmt->getUpdateStatement()->accept(visitor);
    os << ")\n";
    for(size_t i = 0, indentation = printer->getIndentation(); i < indentation; ++i)
        os << "\t";
    auto* loop_body = for_stmt->getLoopBody();
    if(!loop_body->isBlockStatement())
        os << "\t";
    loop_body->accept(visitor);
    if(!loop_body->isBlockStatement())
        os << ";\n";
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const SwitchStatement* switch_stmt)
{
    auto& os = printer->stream();
    os << "switch(";
    switch_stmt->getCondition()->accept(visitor);
    os << ")\n";
    size_t indentation = printer->getIndentation();
    for(size_t i = 0; i < indentation; ++i)
        os << "\t";
    os << "{\n";
    for(size_t i = 0; i < indentation; ++i)
        os << "\t";
    static_cast<AST::VisitorInterface*>(visitor)->visit(switch_stmt->getCases());
    for(size_t i = 0; i < indentation; ++i)
        os << "\t";
    os << "}\n";
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const CaseStatement* case_stmt)
{
    auto& os = printer->stream();
    os << "case ";
    case_stmt->getLabel()->accept(visitor);
    os << ":\n";
    for(size_t i = 0, indentation = printer->getIndentation(); i < indentation; ++i)
        os << "\t";
    auto* statement = case_stmt->getStatement();
    if(!statement->isBlockStatement())
        os << "\t";
    static_cast<AST::VisitorInterface*>(visitor)->visit(statement);
}

void PrintNode(AST::PrinterInfrastructure* printer, const JumpStatement* jump_stmt)
{
    auto& os = printer->stream();
    switch(jump_stmt->getJumpType())
    {
    case JumpStatementType::Continue: os << "continue"; break;
    case JumpStatementType::Break: os << "break"; break;
    default: TGE_ASSERT(false, "Unsupported jump expression");
    }
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ReturnStatement* return_stmt)
{
    auto& os = printer->stream();
    auto* return_expr = return_stmt->getReturnExpression();
    if(return_expr)
    {
        os << "return ";
        return_expr->accept(visitor);
    }
    else
        os << "return";
}

void PrintNode(VisitorInterface* visitor, const Type* type_stmt)
{
    type_stmt->accept(visitor);
}

void PrintNode(AST::PrinterInfrastructure* printer, const Profile* profile)
{
     printer->stream() << "profile";
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ShaderDeclaration* shader)
{
    string shader_type;
    switch(shader->getType())
    {
    case ShaderType::VertexShader:
        shader_type = "vertex"; break;
    case ShaderType::FragmentShader:
        shader_type = "fragment"; break;
    default:
        break;
    }
    shader->printList(visitor, printer, shader_type + " shader");
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const StructType* _struct)
{
    printer->stream() << _struct->getNodeName();
}

void PrintNode(AST::PrinterInfrastructure* printer, const CompiledShader* compiled_shader)
{
    printer->stream() << "compiled shader";
}

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const IfStatement* if_stmt)
{
    auto& os = printer->stream();
    os << "if(";
    if_stmt->getCondition()->accept(visitor);
    os << ")\n";
    size_t indentation = printer->getIndentation();
    auto* true_stmt = if_stmt->getTrueStatement();
    if(*true_stmt)
    {
        for(size_t i = 0; i < indentation; ++i)
            os << "\t";
        if(!true_stmt->isBlockStatement())
            os << "\t";
        true_stmt->accept(visitor);
        if(!true_stmt->isBlockStatement())
            os << ";\n";
    }
    else
        os << ";\n";
    auto* false_stmt = if_stmt->getFalseStatement();
    if(*false_stmt)
    {
        for(size_t i = 0; i < indentation; ++i)
            os << "\t";
        os << "else\n";
        for(size_t i = 0; i < indentation; ++i)
            os << "\t";
        if(!false_stmt->isBlockStatement())
            os << "\t";
        false_stmt->accept(visitor);
        if(!false_stmt->isBlockStatement())
            os << ";\n";
    }
}
}
}
