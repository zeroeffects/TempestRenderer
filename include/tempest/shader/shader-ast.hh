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

#ifndef _GL_EFFECT_AST_HH_
#define _GL_EFFECT_AST_HH_

#include "tempest/utils/memory.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/assert.hh"
#include "tempest/shader/shader-common.hh"
#include "tempest/parser/ast.hh"

namespace Tempest
{
namespace Shader
{
using namespace AST;

enum NodeType
{
    TGE_EFFECT_INTERPOLATION_QUALIFIER = TGE_AST_NODES,
    TGE_EFFECT_PRECISION_QUALIFIER,
    TGE_EFFECT_FUNCTION_DECLARATION,
    TGE_EFFECT_FUNCTION_DEFINITION,
    TGE_EFFECT_FUNCTION_SET,
    TGE_EFFECT_FUNCTION_CALL,
    TGE_EFFECT_DECLARATION,
    TGE_EFFECT_VARIABLE,
    TGE_EFFECT_ARRAY_ELEMENT,
    TGE_EFFECT_STRUCT_MEMBER,
    TGE_EFFECT_TYPE,
    TGE_EFFECT_TYPEDEF,
    TGE_EFFECT_CONSTRUCTOR_CALL,
    TGE_EFFECT_TECHNIQUE,
    TGE_EFFECT_IMPORT,
    TGE_EFFECT_SAMPLER,
    TGE_EFFECT_PASS,
    TGE_EFFECT_UNARY_OPERATOR,
    TGE_EFFECT_BINARY_OPERATOR,
    TGE_EFFECT_TERNARY_IF,
    TGE_EFFECT_INVARIANT_DECLARATION,
    TGE_EFFECT_IF_STATEMENT,
    TGE_EFFECT_WHILE_STATEMENT,
    TGE_EFFECT_FOR_STATEMENT,
    TGE_EFFECT_SWITCH_STATEMENT,
    TGE_EFFECT_CASE_STATEMENT,
    TGE_EFFECT_PARENTHESES_STATEMENT,
    TGE_EFFECT_JUMP_STATEMENT,
    TGE_EFFECT_RETURN_STATEMENT,
    TGE_EFFECT_EXPRESSION,
    TGE_EFFECT_BUFFER
};

enum BinaryOperatorType
{
    TGE_EFFECT_ASSIGN,
    TGE_EFFECT_ADD_ASSIGN,
    TGE_EFFECT_SUB_ASSIGN,
    TGE_EFFECT_MUL_ASSIGN,
    TGE_EFFECT_DIV_ASSIGN,
    TGE_EFFECT_MOD_ASSIGN,
    TGE_EFFECT_BITWISE_AND_ASSIGN,
    TGE_EFFECT_BITWISE_XOR_ASSIGN,
    TGE_EFFECT_BITWISE_OR_ASSIGN,
    TGE_EFFECT_ADD,
    TGE_EFFECT_SUBTRACT,
    TGE_EFFECT_MULTIPLY,
    TGE_EFFECT_DIVIDE,
    TGE_EFFECT_MODULUS,
    TGE_EFFECT_BITWISE_AND,
    TGE_EFFECT_BITWISE_OR,
    TGE_EFFECT_BITWISE_XOR,
    TGE_EFFECT_BITWISE_SHIFT_RIGHT,
    TGE_EFFECT_BITWISE_SHIFT_LEFT,
    TGE_EFFECT_LESS,
    TGE_EFFECT_GREATER,
    TGE_EFFECT_LEQUAL,
    TGE_EFFECT_GEQUAL,
    TGE_EFFECT_OR,
    TGE_EFFECT_AND,
    TGE_EFFECT_XOR,
    TGE_EFFECT_EQUAL,
    TGE_EFFECT_NEQUAL,
    TGE_EFFECT_COMMA
};

enum UnaryOperatorType
{
    TGE_EFFECT_POSITIVE,
    TGE_EFFECT_NEGATE,
    TGE_EFFECT_NOT,
    TGE_EFFECT_COMPLEMENT,
    TGE_EFFECT_PRE_INCR,
    TGE_EFFECT_PRE_DECR,
    TGE_EFFECT_POST_INCR,
    TGE_EFFECT_POST_DECR
};

class Type;
typedef Value<string> Identifier;
template<class TLhs, class TRhs> class Intermediate;
typedef Intermediate<const Type*, AST::Node> Expression;
class FunctionDeclaration;
class FunctionDefinition;
class FunctionSet;
class FunctionCall;
class Declaration;
class Variable;
class ArrayElementVariable;
class MemberVariable;
class Typedef;
class ConstructorCall;
class Import;
class Profile;
class Technique;
class Sampler;
class Shader;
class Pass;
class UnaryOperator;
class BinaryOperator;
class TernaryIf;
class InvariantDeclaration;
class IfStatement;
class WhileStatement;
class ForStatement;
class SwitchStatement;
class CaseStatement;
class Parentheses;
class JumpStatement;
class ReturnStatement;
class VisitorInterface;
class Buffer;
typedef Value<string> Identifier;
typedef Intermediate<const Type*, NodeT<Identifier>> DeclarationInfo;
typedef Intermediate<const Type*, FunctionSet*> FuncDeclarationInfo;
typedef Intermediate<const Type*, NodeT<List>> VarDeclList;
typedef Value<ShaderType> ValueShaderType;

typedef AST::Reference<FunctionSet> FunctionSetRef;
typedef AST::Reference<Variable> VariableRef;
typedef AST::Reference<Type> TypeRef;
typedef AST::Reference<Sampler> SamplerRef;
}

namespace AST
{
using namespace Shader;

TGE_AST_NODE_INFO(FunctionDeclaration, TGE_EFFECT_FUNCTION_DECLARATION, Shader::VisitorInterface)
TGE_AST_NODE_INFO(FunctionDefinition, TGE_EFFECT_FUNCTION_DEFINITION, Shader::VisitorInterface)
TGE_AST_NODE_INFO(FunctionSet, TGE_EFFECT_FUNCTION_SET, Shader::VisitorInterface)
TGE_AST_NODE_INFO(FunctionCall, TGE_EFFECT_FUNCTION_CALL, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Declaration, TGE_EFFECT_DECLARATION, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Variable, TGE_EFFECT_VARIABLE, Shader::VisitorInterface)
TGE_AST_NODE_INFO(ArrayElementVariable, TGE_EFFECT_ARRAY_ELEMENT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(MemberVariable, TGE_EFFECT_STRUCT_MEMBER, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Type, TGE_EFFECT_TYPE, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Typedef, TGE_EFFECT_TYPEDEF, Shader::VisitorInterface)
TGE_AST_NODE_INFO(ConstructorCall, TGE_EFFECT_CONSTRUCTOR_CALL, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Import, TGE_EFFECT_IMPORT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Technique, TGE_EFFECT_TECHNIQUE, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Pass, TGE_EFFECT_PASS, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Sampler, TGE_EFFECT_SAMPLER, Shader::VisitorInterface)
TGE_AST_NODE_INFO(UnaryOperator, TGE_EFFECT_UNARY_OPERATOR, Shader::VisitorInterface)
TGE_AST_NODE_INFO(BinaryOperator, TGE_EFFECT_BINARY_OPERATOR, Shader::VisitorInterface)
TGE_AST_NODE_INFO(TernaryIf, TGE_EFFECT_TERNARY_IF, Shader::VisitorInterface)
TGE_AST_NODE_INFO(InvariantDeclaration, TGE_EFFECT_INVARIANT_DECLARATION, Shader::VisitorInterface)
TGE_AST_NODE_INFO(IfStatement, TGE_EFFECT_IF_STATEMENT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(WhileStatement, TGE_EFFECT_WHILE_STATEMENT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(ForStatement, TGE_EFFECT_FOR_STATEMENT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(SwitchStatement, TGE_EFFECT_SWITCH_STATEMENT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(CaseStatement, TGE_EFFECT_CASE_STATEMENT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Parentheses, TGE_EFFECT_PARENTHESES_STATEMENT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(JumpStatement, TGE_EFFECT_JUMP_STATEMENT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Expression, TGE_EFFECT_EXPRESSION, Shader::VisitorInterface)
TGE_AST_NODE_INFO(ReturnStatement, TGE_EFFECT_RETURN_STATEMENT, Shader::VisitorInterface)
TGE_AST_NODE_INFO(FuncDeclarationInfo, TGE_AST_UNKNOWN, Shader::VisitorInterface)
TGE_AST_NODE_INFO(DeclarationInfo, TGE_AST_UNKNOWN, Shader::VisitorInterface)
TGE_AST_NODE_INFO(VarDeclList, TGE_AST_UNKNOWN, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Value<ShaderType>, TGE_AST_UNKNOWN, Shader::VisitorInterface)
TGE_AST_NODE_INFO(Buffer, TGE_EFFECT_BUFFER, Shader::VisitorInterface)
}

namespace Shader
{
class Driver;

template<class T> struct TypeInfo {};
#define TGE_EFFECT_TYPE_INFO(_type, _type_enum, vis_type) \
    template<> struct TypeInfo<_type> { \
        static const ElementType type_enum = _type_enum; \
        typedef vis_type visitor_type; \
    };

class ScalarType;
class VectorType;
class MatrixType;
class ArrayType;
class Sampler;
class SamplerType;
class Shader;
class Profile;
class CompiledShader;

TGE_EFFECT_TYPE_INFO(ScalarType, ElementType::Scalar, VisitorInterface)
TGE_EFFECT_TYPE_INFO(VectorType, ElementType::Vector, VisitorInterface)
TGE_EFFECT_TYPE_INFO(MatrixType, ElementType::Matrix, VisitorInterface)
TGE_EFFECT_TYPE_INFO(ArrayType, ElementType::Array, VisitorInterface)
TGE_EFFECT_TYPE_INFO(SamplerType, ElementType::Sampler, VisitorInterface)
TGE_EFFECT_TYPE_INFO(Shader, ElementType::Shader, VisitorInterface)
TGE_EFFECT_TYPE_INFO(Profile, ElementType::Profile, VisitorInterface)
TGE_EFFECT_TYPE_INFO(CompiledShader, ElementType::CompiledShader, VisitorInterface)

// The reasoning behind passing this_type is that we can't guarantee that it
// is going to be preserved because it is used as a facade to hide the polymorphism
// and it could potentially be pushed to list. Also, it is ridiculously painfull to
// pass it to any constructor. If you don't agree, feel free to experiment with some
// more elegant solution.
struct TypeImpl
{
public:
    TypeImpl() {}
    virtual ~TypeImpl() {}
    
    virtual string getNodeName() const=0;
    
    virtual bool hasBase(const Type* _type) const=0;

    virtual ElementType getTypeEnum() const=0;

    virtual bool hasImplicitConversionTo(const Type* _type) const=0;

    virtual const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const=0;
    virtual const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const=0;

    virtual const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const=0;
    virtual const Type* getArrayElementType() const=0;
    virtual bool hasValidConstructor(const List* var_list) const=0;
    
    virtual void accept(VisitorInterface* _accept) const=0;
};

template<class T>
struct TypeImplModel: public TypeImpl
{
    template<class... TArgs>
    TypeImplModel(TArgs&&... args)
        :   m_Data(std::forward<TArgs>(args)...) {}
    virtual ~TypeImplModel() {}
        
    virtual string getNodeName() const override { return m_Data.getNodeName(); }

    virtual bool hasBase(const Type* _type) const override { return m_Data.hasBase(_type); }

    virtual ElementType getTypeEnum() const override { return TypeInfo<T>::type_enum; }

    virtual bool hasImplicitConversionTo(const Type* _type) const override { return m_Data.hasImplicitConversionTo(_type); }

    virtual const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const override { return m_Data.binaryOperatorResultType(driver, this_type, binop, operandB); }
    virtual const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const override { return m_Data.unaryOperatorResultType(driver, this_type, uniop); }

    virtual const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const override { return m_Data.getMemberType(driver, this_type, name); }
    virtual const Type* getArrayElementType() const override { return m_Data.getArrayElementType(); }
    virtual bool hasValidConstructor(const List* var_list) const override { return m_Data.hasValidConstructor(var_list); }

    virtual void accept(VisitorInterface* visitor) const { static_cast<typename TypeInfo<T>::visitor_type*>(visitor)->visit(&m_Data); }

    T           m_Data;
};

class Type
{
    TypeImpl* m_Impl;
public:
    Type(TypeImpl* val)
        :   m_Impl(val) {}

    ~Type() { TGE_DEALLOCATE(m_Impl); }
        
    Type(Type&& _type)
        :   m_Impl(_type.m_Impl) 
    {
            TGE_ASSERT(this != &_type, "Misused move constructor");
            _type.m_Impl = nullptr;
    }

    Type& operator=(Type&& val)
    {
        TGE_DEALLOCATE(m_Impl);
        m_Impl = val.m_Impl;
        val.m_Impl = nullptr;
        return *this;
    }
    
    Type(const Type&)=delete;
    Type& operator=(const Type&)=delete;

    string getNodeName() const { return m_Impl->getNodeName(); }

    size_t getNodeType() const { return TGE_EFFECT_TYPE; }
    bool isBlockStatement() const { return false; }

    bool hasBase(const Type* _type) const { return this == _type ? true : m_Impl->hasBase(_type); }

    ElementType getTypeEnum() const { return m_Impl->getTypeEnum(); }

    template<class U>
    U* extract() { TGE_ASSERT(m_Impl->getTypeEnum() == TypeInfo<U>::type_enum, "Unexpected node"); return m_Impl ? &static_cast<TypeImplModel<U>*>(m_Impl)->m_Data : nullptr; }

    template<class U>
    const U* extract() const { TGE_ASSERT(m_Impl->getTypeEnum() == TypeInfo<U>::type_enum, "Unexpected node"); return m_Impl ? &static_cast<TypeImplModel<U>*>(m_Impl)->m_Data : nullptr; }

    void accept(VisitorInterface* visitor) const { m_Impl->accept(visitor); }

    bool hasImplicitConversionTo(const Type* _type) const { return this == _type ? true : m_Impl->hasImplicitConversionTo(_type); }

    const Type* binaryOperatorResultType(Driver& driver, BinaryOperatorType binop, const Type* operandB) const { return m_Impl->binaryOperatorResultType(driver, this, binop, operandB); }
    const Type* unaryOperatorResultType(Driver& driver, UnaryOperatorType uniop) const { return m_Impl->unaryOperatorResultType(driver, this, uniop); }

    const Type* getMemberType(Driver& driver, const string& name) const { return m_Impl->getMemberType(driver, this, name); }
    const Type* getArrayElementType() const { return m_Impl->getArrayElementType(); }
    bool hasValidConstructor(const List* var_list) const { return m_Impl->hasValidConstructor(var_list); }
    NodeT<ConstructorCall> createConstructorCall(AST::Location loc, NodeT<List> arg_list) const { return hasValidConstructor(arg_list.get()) ? CreateNodeTyped<ConstructorCall>(loc, this, std::move(arg_list)) : NodeT<ConstructorCall>(); }
};

template<class T, class... TArgs>
AST::Node CreateTypeNode(TArgs&&... args)
{
    return CreateNode<Type>(TGE_DEFAULT_LOCATION, Type(TGE_ALLOCATE(TypeImplModel<T>)(std::forward<TArgs>(args)...)));
}

class Typedef
{
    string        m_Name;
    Type*         m_Type;
public:
    Typedef(Type* _type, string name);
    ~Typedef();

    string getNodeName() const { return m_Name; }
    
    Type* getType(); // For compilation-related reasons
    const Type* getType() const;

    bool isBlockStatement() const;
};

class Parentheses
{
    AST::Node     m_Node;
public:
    Parentheses(AST::Node _node);
    ~Parentheses();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "()"; }
    
    AST::Node* getExpression();
    const AST::Node* getExpression() const;

    bool isBlockStatement() const;
};

class FunctionCall;

class FunctionDeclaration
{
    const Type*     m_ReturnType;
    string          m_Name;
    NodeT<List>     m_VarList;
public:
    typedef ListIterator<const AST::Node>  const_parameter_iterator;
    typedef ListIterator<AST::Node>        parameter_iterator;

    FunctionDeclaration(const Type* return_type, string name, NodeT<List> var_list);
    ~FunctionDeclaration();

    string getNodeName() const { return m_Name; }
    
    parameter_iterator getParametersBegin();
    parameter_iterator getParametersEnd();

    const_parameter_iterator getParametersBegin() const;
    const_parameter_iterator getParametersEnd() const;

    NodeT<FunctionCall> createFunctionCall(AST::Location loc, NodeT<List> arg_list);
    bool sameParameters(const List* var_list) const;

    const Type* getReturnType() const;
    
    const List* getArgumentExpressions() const { return m_VarList.get(); }
    
    bool isBlockStatement() const;
};

class FunctionDefinition
{
    const FunctionDeclaration* m_Declaration;
    AST::NodeT<List>        m_Body;
public:
    FunctionDefinition(const FunctionDeclaration* decl, AST::NodeT<List> body);
    ~FunctionDefinition();

    const FunctionDeclaration* getDeclaration() const { return m_Declaration; }
    
    List* getBody() { return m_Body.get(); }
    const List* getBody() const { return m_Body.get(); }
    
    string getNodeName() const { return m_Declaration->getNodeName(); }
    
    bool isBlockStatement() const;
};

class FunctionCall
{
    const FunctionDeclaration* m_Function;
    AST::NodeT<List>        m_Args;
public:
    FunctionCall(const FunctionDeclaration* func, AST::NodeT<List> arg_list);
    ~FunctionCall();

    string getNodeName() const { return m_Function->getNodeName(); }
    
    const FunctionDeclaration* getFunction() const;

    List* getArguments();
    const List* getArguments() const;

    bool isBlockStatement() const;
};

class FunctionSet
{
    typedef std::vector<AST::NodeT<FunctionDeclaration>> FunctionList;
    string                      m_Name;
    FunctionList                m_Func;
public:
    FunctionSet(string name);
    ~FunctionSet();

    string getNodeName() const
    {
        return m_Name;
    }
    
    void pushFunction(AST::NodeT<FunctionDeclaration> func);

    FunctionDeclaration* getFunction(const List* var_list);

    FunctionDeclaration* getFunction(size_t idx);
    const FunctionDeclaration* getFunction(size_t idx) const;
    size_t getFunctionCount() const;

    bool isBlockStatement() const;
};

class ConstructorCall
{
    const Type*             m_Type;
    NodeT<List>             m_Args;
public:
    ConstructorCall(const Type* _type, NodeT<List> arg_list);
     ~ConstructorCall();

    string getNodeName() const { return m_Type->getNodeName(); }
     
    const Type* getType() const;

    List* getArguments() { return m_Args.get(); }
    const List* getArguments() const { return m_Args.get(); }
    
    bool isBlockStatement() const;
};

class ScalarType
{
    bool                    m_Integer;
    string                  m_Name;
public:
    ScalarType(bool is_integer, string name);
    ~ScalarType();

    bool isInteger() const { return m_Integer; }
    
    string getNodeName() const { return m_Name; }
    
    bool hasBase(const Type* _type) const { return false; }
    
    bool hasImplicitConversionTo(const Type* _type) const;

    const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const;
    const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const;

    const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const;
    const Type* getArrayElementType() const;
    bool hasValidConstructor(const List* var_list) const;
};

class VectorType
{
    string                  m_Name;
    const Type*             m_Type;
    size_t                  m_VecDim;
public:
    VectorType(const Type* _type, size_t vec_dim, string name);
    ~VectorType();

    string getNodeName() const { return m_Name; }
    
    bool hasBase(const Type* _type) const { return false; }
    
    size_t getDimension() const;
    const Type* getBasicType() const;

    bool hasImplicitConversionTo(const Type* _type) const;
    
    const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const;
    const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const;

    const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const;
    const Type* getArrayElementType() const;
    bool hasValidConstructor(const List* var_list) const;
};

class MatrixType
{
    string                  m_Name;
    size_t                  m_Rows;
    const Type*             m_RowType;
public:
    MatrixType(size_t rows, const Type* row_type, string name);
     ~MatrixType();

    size_t getRows() const;
    size_t getColumns() const;

    string getNodeName() const { return m_Name; }
    
    bool hasBase(const Type* _type) const { return false; }
    
    const Type* getBasicType() const;

    bool hasImplicitConversionTo(const Type* _type) const;
    
    const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const;
    const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const;

    const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const;
    const Type* getArrayElementType() const;
    bool hasValidConstructor(const List* var_list) const;
};

class SamplerType
{
    string      m_Name;
public:
    SamplerType(string name);
    ~SamplerType();
    
    string getNodeName() const { return m_Name; }
    
    bool hasBase(const Type* _type) const { return false; }
    
    bool hasImplicitConversionTo(const Type* _type) const;
    
    const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const;
    const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const;

    const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const;
    const Type* getArrayElementType() const;
    bool hasValidConstructor(const List* var_list) const;
};

class ArrayType
{
    const Type* m_Type;
    AST::Node   m_Size;
public:
    ArrayType(const Type* _type, AST::Node _size);
     ~ArrayType();

    string getNodeName() const { return m_Type->getNodeName() + "[]"; }
     
    const Type* getBasicType() const { return m_Type; }
    
    AST::Node* getSize();
    const AST::Node* getSize() const;

    bool hasBase(const Type* _type) const { return false; }
    
    bool hasImplicitConversionTo(const Type* _type) const;
    
    const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const;
    const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const;

    const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const;
    const Type* getArrayElementType() const;
    bool hasValidConstructor(const List* var_list) const;
};

enum InterpolationQualifier
{
    TGE_EFFECT_DEFAULT_INTERPOLATION,
    TGE_EFFECT_SMOOTH_INTERPOLATION,
    TGE_EFFECT_FLAT_INTERPOLATION,
    TGE_EFFECT_NOPERSPECTIVE_INTERPOLATION
};

enum StorageQualifier
{
    TGE_EFFECT_DEFAULT_STORAGE,
    TGE_EFFECT_CONST_STORAGE,
    TGE_EFFECT_IN_STORAGE,
    TGE_EFFECT_CENTROID_IN_STORAGE,
    TGE_EFFECT_SAMPLE_IN_STORAGE,
    TGE_EFFECT_OUT_STORAGE,
    TGE_EFFECT_CENTROID_OUT_STORAGE,
    TGE_EFFECT_SAMPLE_OUT_STORAGE,
    TGE_EFFECT_INOUT_STORAGE
};

enum PrecisionQualifier
{
    TGE_EFFECT_HIGH_PRECISION,
    TGE_EFFECT_MEDIUM_PRECISION,
    TGE_EFFECT_LOW_PRECISION
};

enum JumpStatementType
{
    TGE_EFFECT_CONTINUE_STATEMENT,
    TGE_EFFECT_BREAK_STATEMENT
    //TGE_EFFECT_GOTO_STATEMENT
};

class Variable
{
    string                          m_Name;
    InterpolationQualifier          m_Interpolation;
    StorageQualifier                m_Storage;
    bool                            m_Invariant;
    InvariantDeclaration*           m_InvariantDecl;
    AST::NodeT<List>                m_Layout;
    const Type*                     m_Type;
public:
    Variable(StorageQualifier _storage, const Type* _type, string name);
    Variable(const Type* type, string name);
     ~Variable();

    void setLayout(AST::NodeT<List> _list);
    void setInterpolation(InterpolationQualifier ipl);
    void setStorage(StorageQualifier storage);
    void setInvariant(bool val);
    void setInvariantDeclaration(InvariantDeclaration* inv_dcl);
    List* getLayout();
    const List* getLayout() const;
    InterpolationQualifier getInterpolation() const;
    StorageQualifier getStorage() const;
    bool getInvariant() const;

    string getNodeName() const { return m_Name; }
    
    const Type* getType() const;
    void setType(const Type* _type);
    
    bool isBlockStatement() const;
};

class Declaration
{
    AST::Node                 m_Variables;
public:
    Declaration(AST::Node var);
     ~Declaration();

    string getNodeName() const { TGE_ASSERT(false, "That should not get called"); return ""; }
     
    AST::Node* getVariables();
    const AST::Node* getVariables() const;

    bool isBlockStatement() const;
};

void PrintType(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Variable* var);

class MemberVariable
{
    Variable                  m_Variable;
    AST::Node                 m_Parent;
public:
    MemberVariable(AST::Node parent, const Type* type, string name);
    ~MemberVariable();

    string getNodeName() const { return m_Variable.getNodeName(); }
    
    bool isBlockStatement() const;
    
    Variable* getVariable() { return &m_Variable; }
    const Variable* getVariable() const { return &m_Variable; }
    
    AST::Node* getParent() { return &m_Parent; }
    const AST::Node* getParent() const { return &m_Parent; }
};

class ArrayElementVariable
{
    Variable                  m_Variable;
    AST::Node                 m_Parent;
    AST::Node                 m_Expr;
public:
    ArrayElementVariable(AST::Node parent, const Type* type, AST::Node expr);
    ~ArrayElementVariable();

    string getNodeName() const { return m_Variable.getNodeName(); }
    
    AST::Node* getParent() { return &m_Parent; }
    const AST::Node* getParent() const { return &m_Parent; }
    
    AST::Node* getIndex() { return &m_Expr; }
    const AST::Node* getIndex() const { return &m_Expr; }
    
    bool isBlockStatement() const;
};

class InvariantDeclaration
{
    Variable*                 m_Variable;
public:
    InvariantDeclaration(Variable* var);
    ~InvariantDeclaration();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "invariant"; }
    
    Variable* getVariable() { return m_Variable; }
    const Variable* getVariable() const { return m_Variable; }
    
    bool isBlockStatement() const;
};

class BinaryOperator
{
    BinaryOperatorType  m_Type;
    AST::Node           m_First,
                        m_Second;
public:
    BinaryOperator(BinaryOperatorType _type, AST::Node _first, AST::Node _second);
    ~BinaryOperator();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "binary operation"; }
    
    BinaryOperatorType getOperation() const;

    AST::Node* getLHSOperand();
    const AST::Node* getLHSOperand() const;
    AST::Node* getRHSOperand();
    const AST::Node* getRHSOperand() const;
    bool isBlockStatement() const;
};

class UnaryOperator
{
    UnaryOperatorType   m_Type;
    AST::Node           m_Operand;
public:
    UnaryOperator(UnaryOperatorType _type, AST::Node _operand);
    ~UnaryOperator();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "unary operation"; }

    AST::Node* getOperand();
    const AST::Node* getOperand() const;
    
    UnaryOperatorType getOperation() const { return m_Type; }
    
    bool isBlockStatement() const;
};

class TernaryIf
{
    AST::Node          m_Condition,
                       m_TrueExpr,
                       m_FalseExpr;
public:
    TernaryIf(AST::Node cond, AST::Node true_expr, AST::Node false_expr);
    ~TernaryIf();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "?"; }

    AST::Node* getCondition() { return &m_Condition; }
    const AST::Node* getCondition() const { return &m_Condition; }
    
    AST::Node* getTrueExpression() { return &m_TrueExpr; }
    const AST::Node* getTrueExpression() const { return &m_TrueExpr; }
    
    AST::Node* getFalseExpression() { return &m_FalseExpr; }
    const AST::Node* getFalseExpression() const { return &m_FalseExpr; }
    
    bool isBlockStatement() const;
};

class IfStatement
{
    AST::Node          m_Condition,
                       m_TrueStatement,
                       m_FalseStatement;
public:
    IfStatement(AST::Node condition_statement, AST::Node true_statement, AST::Node false_statement = AST::Node());
    ~IfStatement();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "if"; }

    AST::Node* getCondition() { return &m_Condition; }
    const AST::Node* getCondition() const { return &m_Condition; }
    
    AST::Node* getTrueStatement() { return &m_TrueStatement; }
    const AST::Node* getTrueStatement() const { return &m_TrueStatement; }
    
    AST::Node* getFalseStatement() { return &m_FalseStatement; }
    const AST::Node* getFalseStatement() const { return &m_FalseStatement; }
    
    bool isBlockStatement() const;
};

class WhileStatement
{
    AST::Node          m_Condition,
                       m_Statement;
    bool               m_IsDoWhile;
public:
    WhileStatement(AST::Node condition_statement, AST::Node statement, bool do_while = false);
    ~WhileStatement();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "while"; }

    AST::Node* getCondition() { return &m_Condition; }
    const AST::Node* getCondition() const { return &m_Condition; }
    
    AST::Node* getLoopBody() { return &m_Statement; }
    const AST::Node* getLoopBody() const { return &m_Statement; }
    
    bool isDoWhile() const { return m_IsDoWhile; }
    
    bool isBlockStatement() const;
};

class ForStatement
{
    AST::Node          m_Init,
                       m_Condition,
                       m_Update,
                       m_Statement;
public:
    ForStatement(AST::Node init_statement, AST::Node condition_statement, AST::Node loop_statement, AST::Node statement);
    ~ForStatement();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "for"; }

    AST::Node* getInitStatement() { return &m_Init; }
    const AST::Node* getInitStatement() const { return &m_Init; }
    
    AST::Node* getCondition() { return &m_Condition; }
    const AST::Node* getCondition() const { return &m_Condition; }
    
    AST::Node* getUpdateStatement() { return &m_Update; }
    const AST::Node* getUpdateStatement() const { return &m_Update; }
    
    AST::Node* getLoopBody() { return &m_Statement; }
    const AST::Node* getLoopBody() const { return &m_Statement; }
    
    bool isBlockStatement() const;
};

class SwitchStatement
{
    AST::Node          m_Condition;
    NodeT<List>        m_Cases;
public:
    SwitchStatement(AST::Node condition_statement, NodeT<List> cases);
    ~SwitchStatement();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "switch"; }

    AST::Node* getCondition() { return &m_Condition; }
    const AST::Node* getCondition() const { return &m_Condition; }
    
    List* getCases() { return m_Cases.get(); }
    const List* getCases() const { return m_Cases.get(); }
    
    bool isBlockStatement() const;
};

class CaseStatement
{
    AST::Node          m_Label;
    NodeT<List>        m_Statement;
public:
    CaseStatement(AST::Node label, NodeT<List> statement);
    ~CaseStatement();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "case"; }

    AST::Node* getLabel() { return &m_Label; }
    const AST::Node* getLabel() const { return &m_Label; }
    
    List* getStatement() { return m_Statement.get(); }
    const List* getStatement() const { return m_Statement.get(); }
    
    bool isBlockStatement() const;
};

class JumpStatement
{
    JumpStatementType  m_JumpType;
public:
    JumpStatement(JumpStatementType jump_type);
    ~JumpStatement();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "jump"; }
    
    JumpStatementType getJumpType() const { return m_JumpType; }

    bool isBlockStatement() const;
};

class ReturnStatement
{
    AST::Node          m_ReturnExpression;
public:
    ReturnStatement(AST::Node retexpr=AST::Node());
    ~ReturnStatement();

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "return"; }
    
    AST::Node* getReturnExpression() { return &m_ReturnExpression; }
    const AST::Node* getReturnExpression() const { return &m_ReturnExpression; }
    
    bool isBlockStatement() const { return false; }
};

template<class T>
struct AddConst
{
	typedef const T result_type;
};

template<class T1, class T2>
class Intermediate
{
    T1                 m_Value1;
    T2                 m_Value2;
public:
	typedef T1									first_value_type;
	typedef T2									second_value_type;
	typedef typename AddConst<T1>::result_type  first_value_const_type;
	typedef typename AddConst<T2>::result_type	second_value_const_type;

    Intermediate(T1 t1, T2 t2)
        :   m_Value1(std::move(t1)),
            m_Value2(std::move(t2)) {}
     ~Intermediate() {}

    string getNodeName() const { TGE_ASSERT(false, "Should not get called"); return "intermediate"; }
     
    first_value_type& getFirst() { return m_Value1; }
	first_value_const_type& getFirst() const { return m_Value1; }
    second_value_type& getSecond() { return m_Value2; }
	second_value_const_type& getSecond() const { return m_Value2; }
    void setFirst(T1&& t) { m_Value1 = std::move(t); }
    void setSecond(T2&& t) { m_Value2 = std::move(t); }

    bool isBlockStatement() const { return false; }
};

class Profile
{
public:
    Profile();
    ~Profile();

    bool hasValidConstructor(const List& var_list) const;
    
    string getNodeName() const { return "profile"; }
    
    bool hasBase(const Type* _type) const { return false; }
    
    bool hasImplicitConversionTo(const Type* _type) const;
    
    const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const;
    const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const;

    const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const;
    const Type* getArrayElementType() const;
    bool hasValidConstructor(const List* var_list) const;
};

class Technique: public AST::NamedList<Technique>
{
public:
    Technique(string name, NodeT<List> body);
     ~Technique();
};

class Sampler: public AST::NamedList<Sampler>
{
public:
    Sampler(string name, NodeT<List> body);
     ~Sampler();
};

class Import: public AST::NamedList<Import>
{
public:
    Import(string name, NodeT<List> body);
     ~Import();
};

class Buffer: public AST::NamedList<Buffer>
{
    BufferType         m_BufferType;
public:
    Buffer(string name, NodeT<List> body);
     ~Buffer();
     
    void setBufferType(BufferType buffer_type) { m_BufferType = buffer_type; }
    BufferType getBufferType() const { return m_BufferType; }
};

class Shader: public AST::NamedList<Shader>
{
    ShaderType          m_Type;
public:
    Shader(ShaderType _type, string name, NodeT<List> body);
    ~Shader();

    ShaderType getType() const;
    
    bool hasBase(const Type* _type) const;
    
    bool hasImplicitConversionTo(const Type* _type) const;
    
    const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const;
    const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const;

    const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const;
    const Type* getArrayElementType() const;
    bool hasValidConstructor(const List* var_list) const;
};

class CompiledShader
{
public:
    CompiledShader();
     ~CompiledShader();
    
    string getNodeName() const { return "compiled shader"; }
     
    bool hasBase(const Type* _type) const { return false; }
    
    bool hasImplicitConversionTo(const Type* _type) const;
    
    const Type* binaryOperatorResultType(Driver& driver, const Type* this_type, BinaryOperatorType binop, const Type* operandB) const;
    const Type* unaryOperatorResultType(Driver& driver, const Type* this_type, UnaryOperatorType uniop) const;

    const Type* getMemberType(Driver& driver, const Type* this_type, const string& name) const;
    const Type* getArrayElementType() const;
    bool hasValidConstructor(const List* var_list) const;
};

class Pass: public AST::NamedList<Pass>
{
public:
    Pass(string name, NodeT<List> body);
    ~Pass();
};

class VisitorInterface: public AST::VisitorInterface
{
public:
    VisitorInterface() {}
    virtual ~VisitorInterface() {}
    
    virtual void visit(const Typedef*)=0;
    virtual void visit(const Parentheses*)=0;
    virtual void visit(const FunctionDeclaration*)=0;
    virtual void visit(const FunctionDefinition*)=0;
    virtual void visit(const FunctionCall*)=0;
    virtual void visit(const FunctionSet*)=0;
    virtual void visit(const ConstructorCall*)=0;
    virtual void visit(const ScalarType*)=0;
    virtual void visit(const VectorType*)=0;
    virtual void visit(const MatrixType*)=0;
    virtual void visit(const SamplerType*)=0;
    virtual void visit(const ArrayType*)=0;
    virtual void visit(const Variable*)=0;
    virtual void visit(const Declaration*)=0;
    virtual void visit(const MemberVariable*)=0;
    virtual void visit(const ArrayElementVariable*)=0;
    virtual void visit(const InvariantDeclaration*)=0;
    virtual void visit(const BinaryOperator*)=0;
    virtual void visit(const UnaryOperator*)=0;
    virtual void visit(const TernaryIf*)=0;
    virtual void visit(const WhileStatement*)=0;
    virtual void visit(const ForStatement*)=0;
    virtual void visit(const SwitchStatement*)=0;
    virtual void visit(const CaseStatement*)=0;
    virtual void visit(const JumpStatement*)=0;
    virtual void visit(const ReturnStatement*)=0;
    virtual void visit(const Expression*)=0;
    virtual void visit(const Profile*)=0;
    virtual void visit(const Technique*)=0;
    virtual void visit(const Sampler*)=0;
    virtual void visit(const Import*)=0;
    virtual void visit(const Shader*)=0;
    virtual void visit(const CompiledShader*)=0;
    virtual void visit(const Pass*)=0;
    virtual void visit(const IfStatement*)=0;
    virtual void visit(const Buffer*)=0;
    // Some types that should not appear in AST
    virtual void visit(const FuncDeclarationInfo*)=0;
    virtual void visit(const DeclarationInfo*)=0;
    virtual void visit(const VarDeclList*)=0;
    virtual void visit(const Value<ShaderType>*)=0;
    virtual void visit(const Type* type_stmt)=0;
};

void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Typedef* _typedef) ;
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Parentheses* parentheses) ;
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const FunctionDeclaration* func_decl) ;
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const FunctionDefinition* func_def);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const FunctionCall* func_call);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ConstructorCall* const_call);
void PrintNode(AST::PrinterInfrastructure* printer, const ScalarType* scalar_type);
void PrintNode(AST::PrinterInfrastructure* printer, const VectorType* vector_type);
void PrintNode(AST::PrinterInfrastructure* printer, const MatrixType* matrix_type);
void PrintNode(AST::PrinterInfrastructure* printer, const SamplerType* sampler_type);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ArrayType* array_type);
void PrintNode(AST::PrinterInfrastructure* printer, const Variable* variable);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Declaration* decl);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const MemberVariable* mem_var);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ArrayElementVariable* array_elem);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const InvariantDeclaration* invar_decl);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const BinaryOperator* binop);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const UnaryOperator* unaryop);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const TernaryIf* ternary_if);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const WhileStatement* while_stmt);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ForStatement* for_stmt);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const SwitchStatement* switch_stmt);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const CaseStatement* case_stmt);
void PrintNode(AST::PrinterInfrastructure* printer, const JumpStatement* jump_stmt);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const ReturnStatement* return_stmt);
void PrintNode(AST::PrinterInfrastructure* printer, const Profile* profile_stmt);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Shader* shader_stmt);
void PrintNode(AST::PrinterInfrastructure* printer, const CompiledShader* compiled_stmt);
void PrintNode(VisitorInterface* visitor, const Type* type_stmt);
void PrintNode(VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const IfStatement* if_stmt);

/*! \brief Outputs the content of the AST graph in formatted fashion.
 *
 *  The main purpose of this class is to output information about some elements of the AST graph
 *  when error gets encountered. There are different classes for outputting the graph in some
 *  specific shading language.
 */
class Printer: public VisitorInterface
{
    AST::PrinterInfrastructure m_Printer;
public:
    Printer(std::ostream& os, size_t flags);
    virtual ~Printer();

    std::ostream& stream() { return m_Printer.stream(); }

    virtual void visit(const AST::Value<float>* value) override { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::Value<int>* value) override { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::Value<unsigned>* value) override { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::Value<bool>* value) override { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::Value<string>* value) override { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::ListElement* lst) override { AST::PrintNode(this, &m_Printer, lst); }
    virtual void visit(const AST::Block* _block) override { AST::PrintNode(this, &m_Printer, _block);}
    virtual void visit(const AST::StringLiteral* value) override { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const Typedef* _typedef) override { PrintNode(this, &m_Printer, _typedef);}
    virtual void visit(const Parentheses* parentheses) override { PrintNode(this, &m_Printer, parentheses);}
    virtual void visit(const FunctionDeclaration* func_decl) override { PrintNode(this, &m_Printer, func_decl); }
    virtual void visit(const FunctionDefinition* func_def) override { PrintNode(this, &m_Printer, func_def); }
    virtual void visit(const FunctionCall* func_call) override { PrintNode(this, &m_Printer, func_call); }
    virtual void visit(const ConstructorCall* constructor) override { PrintNode(this, &m_Printer, constructor); }
    virtual void visit(const ScalarType* scalar_type) override { PrintNode(&m_Printer, scalar_type); }
    virtual void visit(const VectorType* vector_type) override { PrintNode(&m_Printer, vector_type); }
    virtual void visit(const MatrixType* matrix_type) override { PrintNode(&m_Printer, matrix_type); }
    virtual void visit(const SamplerType* sampler_type) override { PrintNode(&m_Printer, sampler_type); }
    virtual void visit(const ArrayType* array_type) override { PrintNode(this, &m_Printer, array_type); }
    virtual void visit(const Variable* var) override { PrintNode(&m_Printer, var); }
    virtual void visit(const Declaration* decl) override { PrintNode(this, &m_Printer, decl); }
    virtual void visit(const MemberVariable* mem_var) override { PrintNode(this, &m_Printer, mem_var); }
    virtual void visit(const ArrayElementVariable* array_elem) override { PrintNode(this, &m_Printer, array_elem); }
    virtual void visit(const InvariantDeclaration* invar_decl) override { PrintNode(this, &m_Printer, invar_decl); }
    virtual void visit(const BinaryOperator* binop) override { PrintNode(this, &m_Printer, binop); }
    virtual void visit(const UnaryOperator* unaryop) override { PrintNode(this, &m_Printer, unaryop); }
    virtual void visit(const TernaryIf* ternary_if) override { PrintNode(this, &m_Printer, ternary_if); }
    virtual void visit(const WhileStatement* while_stmt) override { PrintNode(this, &m_Printer, while_stmt); }
    virtual void visit(const ForStatement* for_stmt) override { PrintNode(this, &m_Printer, for_stmt); }
    virtual void visit(const SwitchStatement* switch_stmt) override { PrintNode(this, &m_Printer, switch_stmt); }
    virtual void visit(const CaseStatement* case_stmt) override { PrintNode(this, &m_Printer, case_stmt); }
    virtual void visit(const JumpStatement* jump_stmt) override { PrintNode(&m_Printer, jump_stmt); }
    virtual void visit(const ReturnStatement* return_stmt) override { PrintNode(this, &m_Printer, return_stmt); }
    virtual void visit(const Profile* _profile) override { PrintNode(&m_Printer, _profile); }
    virtual void visit(const Technique* _technique) override { _technique->printList(this, &m_Printer, "technique"); }
    virtual void visit(const Sampler* _sampler) override { _sampler->printList(this, &m_Printer, "sampler"); }
    virtual void visit(const Import* _import) override { _import->printList(this, &m_Printer, "import"); }
    virtual void visit(const Shader* _shader) override { PrintNode(this, &m_Printer, _shader); }
    virtual void visit(const CompiledShader* compiled_shader) override { PrintNode(&m_Printer, compiled_shader); }
    virtual void visit(const Pass* _pass) override { _pass->printList(this, &m_Printer, "pass"); }
    virtual void visit(const IfStatement* if_stmt) override { PrintNode(this, &m_Printer, if_stmt); }
    virtual void visit(const Type* type_stmt) override { PrintNode(this, type_stmt); }
    virtual void visit(const Buffer* buffer) override { buffer->printList(this, &m_Printer, "buffer");}
    // Some types that should not appear in AST
    virtual void visit(const FunctionSet*) override { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Expression*) override { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const FuncDeclarationInfo*) override { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const DeclarationInfo*) override { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const VarDeclList*) override { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Value<ShaderType>*) override { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
};
}
}

#endif /* _GL_EFFECT_AST_HH_ */
