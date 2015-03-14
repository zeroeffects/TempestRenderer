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

#include "tempest/shader/dx-shader-generator.hh"
#include "tempest/parser/file-loader.hh"
#include "tempest/shader/shader-ast.hh"
#include "tempest/shader/shader-driver.hh"
#include "tempest/shader/shader-common.hh"
#include "tempest/shader/shader-convert-common.hh"
#include "tempest/parser/ast.hh"

#include <unordered_map>
#include <algorithm>
#include <stack>

namespace Tempest
{
namespace DXFX
{
typedef std::unordered_map<string, string> SamplerTextureAssociation;
typedef std::vector<string> ShaderSignature;

void PrintBuffer(AST::VisitorInterface* visitor, AST::PrinterInfrastructure* m_Printer, const Shader::Buffer* buffer);

class ShaderPrinter: public Shader::VisitorInterface
{
    AST::PrinterInfrastructure m_Printer;
    
    Shader::Driver&            m_Driver;

    const char*                m_Filename;
    bool                       m_Valid;
    
    SamplerTextureAssociation  m_SamplerAssoc;
    ShaderSignature            m_InputSignature;
    ShaderSignature            m_OutputSignature;
    
    const string*              m_Options;
    size_t                     m_OptionCount;

    typedef bool (ShaderPrinter::*TranslationFunction)(const Shader::FunctionCall* func_call);
    std::unordered_map<string, TranslationFunction> m_FunctionTranslator;
public:
    ShaderPrinter(Shader::Driver& driver, const char* filename, std::ostream& os, const string* opts, size_t opts_count, uint32 flags);
    virtual ~ShaderPrinter();

    void setSamplerTextureAssociation(string texture, string sampler)
    {
        m_SamplerAssoc[texture] = sampler;
    }
    
    void setInputSignature(ShaderSignature signature) { m_InputSignature = signature; }
    void setOutputSignature(ShaderSignature signature) { m_OutputSignature = signature; }
    
    void setIndentation(size_t indentation) { m_Printer.setIndentation(indentation); }
    
    std::ostream& stream() { return m_Printer.stream(); }
    AST::PrinterInfrastructure* getPrinter() { return &m_Printer; }

    bool isValid() const { return m_Valid; }
    
    const char* getFilename() const { return m_Filename; }

    virtual void visit(const Location& loc) final { AST::PrintLocation(&m_Printer, loc, m_Filename); }
    virtual void visit(const AST::Value<float>* value) final { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::Value<int>* value) final { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::Value<unsigned>* value) final { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::Value<bool>* value) final { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::Value<string>* value) final { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const AST::ListElement* lst) final { AST::PrintNode(this, &m_Printer, lst); }
    virtual void visit(const AST::Block* _block) final { AST::PrintNode(this, &m_Printer, _block);}
    virtual void visit(const AST::StringLiteral* value) final { AST::PrintNode(&m_Printer, value); }
    virtual void visit(const Shader::Typedef* _typedef) final { Shader::PrintNode(this, &m_Printer, _typedef);}
    virtual void visit(const Shader::Parentheses* parentheses) final { Shader::PrintNode(this, &m_Printer, parentheses);}
    virtual void visit(const Shader::FunctionDeclaration* func_decl) final { Shader::PrintNode(this, &m_Printer, func_decl); }
    virtual void visit(const Shader::FunctionDefinition* func_def) final { Shader::PrintNode(this, &m_Printer, func_def); }
    virtual void visit(const Shader::FunctionCall* func_call) final;
    virtual void visit(const Shader::ConstructorCall* constructor) final { Shader::PrintNode(this, &m_Printer, constructor); }
    virtual void visit(const Shader::ScalarType* scalar_type) final { Shader::PrintNode(&m_Printer, scalar_type); }
    virtual void visit(const Shader::VectorType* vector_type) final;
    virtual void visit(const Shader::MatrixType* matrix_type) final;
    virtual void visit(const Shader::SamplerType* sampler_type) final { Shader::PrintNode(&m_Printer, sampler_type); }
    virtual void visit(const Shader::ArrayType* array_type) final { Shader::PrintNode(this, &m_Printer, array_type); }
    virtual void visit(const Shader::Variable* var) final;
    virtual void visit(const Shader::Declaration* decl) final { Shader::PrintNode(this, &m_Printer, decl); }
    virtual void visit(const Shader::MemberVariable* mem_var) final { Shader::PrintNode(this, &m_Printer, mem_var); }
    virtual void visit(const Shader::ArrayElementVariable* array_elem) final { Shader::PrintNode(this, &m_Printer, array_elem); }
    virtual void visit(const Shader::InvariantDeclaration* invar_decl) final;
    virtual void visit(const Shader::BinaryOperator* binop) final;
    virtual void visit(const Shader::UnaryOperator* unaryop) final { Shader::PrintNode(this, &m_Printer, unaryop); }
    virtual void visit(const Shader::TernaryIf* ternary_if) final { Shader::PrintNode(this, &m_Printer, ternary_if); }
    virtual void visit(const Shader::WhileStatement* while_stmt) final { Shader::PrintNode(this, &m_Printer, while_stmt); }
    virtual void visit(const Shader::ForStatement* for_stmt) final { Shader::PrintNode(this, &m_Printer, for_stmt); }
    virtual void visit(const Shader::SwitchStatement* switch_stmt) final { Shader::PrintNode(this, &m_Printer, switch_stmt); }
    virtual void visit(const Shader::CaseStatement* case_stmt) final { Shader::PrintNode(this, &m_Printer, case_stmt); }
    virtual void visit(const Shader::JumpStatement* jump_stmt) final { Shader::PrintNode(&m_Printer, jump_stmt); }
    virtual void visit(const Shader::ReturnStatement* return_stmt) final { Shader::PrintNode(this, &m_Printer, return_stmt); }
    virtual void visit(const Shader::Import* _import) final { _import->printList(this, &m_Printer, "import"); }
    virtual void visit(const Shader::ShaderDeclaration* _shader) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::OptionsDeclaration* _opt_decl) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Optional* _opt) final { PrintOptional(this, &m_Printer, _opt, m_Options, m_OptionCount); }
    virtual void visit(const Shader::Option* _opt) final { PrintNode(&m_Printer, _opt); }
    virtual void visit(const Shader::StructType* _struct) final { Shader::PrintNode(this, &m_Printer, _struct); }
    virtual void visit(const Shader::IfStatement* if_stmt) final { Shader::PrintNode(this, &m_Printer, if_stmt); }
    virtual void visit(const Shader::Type* type_stmt) final { Shader::PrintNode(this, type_stmt); }
    virtual void visit(const Shader::Buffer* buffer) final { PrintBuffer(this, &m_Printer, buffer); }
    // Some types that should not appear in AST
    virtual void visit(const Shader::IntermFuncNode*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FunctionSet*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Expression*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FuncDeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::DeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::VarDeclList*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Value<Shader::ShaderType>*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    
private:
    bool TranslateTexelFetch(const Shader::FunctionCall* func_call);
};

class TypeDeducer: public Shader::VisitorInterface
{
    Shader::Driver&                         m_Driver;
    std::stack<const Shader::Type*>         m_TypeStack;
public:
    TypeDeducer(Shader::Driver& driver)
        :   m_Driver(driver) {}

    const Shader::Type* getResult() const { return m_TypeStack.top(); }

    virtual void visit(const Location& loc) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const AST::Value<float>* value) final;
    virtual void visit(const AST::Value<int>* value) final;
    virtual void visit(const AST::Value<unsigned>* value) final;
    virtual void visit(const AST::Value<bool>* value) final;
    virtual void visit(const AST::Value<string>* value) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const AST::ListElement* lst) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const AST::Block* _block) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const AST::StringLiteral* value) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Typedef* _typedef) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Parentheses* parentheses) final;
    virtual void visit(const Shader::FunctionDeclaration* func_decl) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FunctionDefinition* func_def) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FunctionCall* func_call) final;
    virtual void visit(const Shader::ConstructorCall* constructor) final;
    virtual void visit(const Shader::ScalarType* scalar_type) final { TGE_ASSERT(false, "Types should not be entered. Catching up a variable should be enough"); }
    virtual void visit(const Shader::VectorType* vector_type) final { TGE_ASSERT(false, "Types should not be entered. Catching up a variable should be enough"); }
    virtual void visit(const Shader::MatrixType* matrix_type) final { TGE_ASSERT(false, "Types should not be entered. Catching up a variable should be enough"); }
    virtual void visit(const Shader::SamplerType* sampler_type) final { TGE_ASSERT(false, "Types should not be entered. Catching up a variable should be enough"); }
    virtual void visit(const Shader::ArrayType* array_type) final { TGE_ASSERT(false, "Types should not be entered.Catching up a variable should be enough"); }
    virtual void visit(const Shader::Variable* var) final;
    virtual void visit(const Shader::Declaration* decl) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::MemberVariable* mem_var) final;
    virtual void visit(const Shader::ArrayElementVariable* array_elem) final;
    virtual void visit(const Shader::InvariantDeclaration* invar_decl) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::BinaryOperator* binop) final;
    virtual void visit(const Shader::UnaryOperator* unaryop) final;
    virtual void visit(const Shader::TernaryIf* ternary_if) final;
    virtual void visit(const Shader::WhileStatement* while_stmt) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::ForStatement* for_stmt) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::SwitchStatement* switch_stmt) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::CaseStatement* case_stmt) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::JumpStatement* jump_stmt) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::ReturnStatement* return_stmt) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Import* _import) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::ShaderDeclaration* _shader) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::OptionsDeclaration* _opt_decl) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Optional* _opt) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Option* _opt) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::StructType* _struct) final { TGE_ASSERT(false, "Types should not be entered.Catching up a variable should be enough"); }
    virtual void visit(const Shader::IfStatement* if_stmt) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Type* type_stmt) final { TGE_ASSERT(false, "Types should not be entered.Catching up a variable should be enough"); }
    virtual void visit(const Shader::Buffer* buffer) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    // Some types that should not appear in AST
    virtual void visit(const Shader::IntermFuncNode*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FunctionSet*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Expression*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FuncDeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::DeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::VarDeclList*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Value<Shader::ShaderType>*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
};

void TypeDeducer::visit(const AST::Value<float>* value)
{
    m_TypeStack.push(m_Driver.find("float"));
}

void TypeDeducer::visit(const AST::Value<int>* value)
{
    m_TypeStack.push(m_Driver.find("int"));
}

void TypeDeducer::visit(const AST::Value<unsigned>* value)
{
    m_TypeStack.push(m_Driver.find("uint"));
}

void TypeDeducer::visit(const AST::Value<bool>* value)
{
    m_TypeStack.push(m_Driver.find("bool"));
}

void TypeDeducer::visit(const Shader::Parentheses* parentheses)
{
    parentheses->getExpression()->accept(this);
}

void TypeDeducer::visit(const Shader::FunctionCall* func_call)
{
    m_TypeStack.push(func_call->getFunction()->getReturnType());
}

void TypeDeducer::visit(const Shader::ConstructorCall* constructor)
{
    m_TypeStack.push(constructor->getType());
}

void TypeDeducer::visit(const Shader::Variable* var)
{
    m_TypeStack.push(var->getType());
}

void TypeDeducer::visit(const Shader::MemberVariable* mem_var)
{
    m_TypeStack.push(mem_var->getVariable()->getType());
}

void TypeDeducer::visit(const Shader::ArrayElementVariable* array_elem)
{
    m_TypeStack.push(array_elem->getVariable()->getType());
}

void TypeDeducer::visit(const Shader::BinaryOperator* binop)
{
    binop->getLHSOperand()->accept(this);
    binop->getRHSOperand()->accept(this);
    auto* rhs_type = m_TypeStack.top();
    m_TypeStack.pop();
    auto* lhs_type = m_TypeStack.top();
    m_TypeStack.pop();
    m_TypeStack.push(lhs_type->binaryOperatorResultType(m_Driver, binop->getOperation(), rhs_type));
}

void TypeDeducer::visit(const Shader::UnaryOperator* unaryop)
{
    unaryop->getOperand()->accept(this);
    auto* type = m_TypeStack.top();
    m_TypeStack.pop();
    m_TypeStack.push(type->unaryOperatorResultType(m_Driver, unaryop->getOperation()));
}

void TypeDeducer::visit(const Shader::TernaryIf* ternary_if)
{
    ternary_if->getTrueExpression()->accept(this);
    // And it should contain the right type.
}

const Shader::Type* DeduceType(Shader::Driver& driver, const AST::Node* _node)
{
    TypeDeducer td(driver);
    _node->accept(&td);
    return td.getResult();
}

void ShaderPrinter::visit(const Shader::BinaryOperator* binop)
{
    if(binop->getOperation() == Shader::TGE_EFFECT_MULTIPLY &&
       binop->getLHSOperand())
    {
        auto* lhs_operand = binop->getLHSOperand();
        auto* rhs_operand = binop->getRHSOperand();
        auto* lhs_type = DeduceType(m_Driver, lhs_operand);
        auto* rhs_type = DeduceType(m_Driver, rhs_operand);
        if(lhs_type->getTypeEnum() == Shader::ElementType::Matrix ||
           rhs_type->getTypeEnum() == Shader::ElementType::Matrix)
        {
            auto& os = m_Printer.stream();
            os << "mul(";
            lhs_operand->accept(this);
            os << ", ";
            rhs_operand->accept(this);
            os << ")";
            return;
        }
    }
    Shader::PrintNode(this, &m_Printer, binop);
}

void PrintBuffer(AST::VisitorInterface* visitor, AST::PrinterInfrastructure* m_Printer, const Shader::Buffer* buffer)
{
    auto& os = m_Printer->stream();
    switch(buffer->getBufferType())
    {
    case Shader::BufferType::Constant:
    {
        os << "cbuffer " << buffer->getNodeName() << " {\n";
            visitor->visit(buffer->getBody());
        os << "};\n";
    } break;
    case Shader::BufferType::Regular:
    {
        os << "struct " << buffer->getNodeName() << "Struct__" << " {\n";
            visitor->visit(buffer->getBody());
        os << "};\n"
              "StructuredBuffer<" << buffer->getNodeName() << "Struct__> " << buffer->getNodeName() << ";";

    } break;
    case Shader::BufferType::Resource:
    {
        TGE_ASSERT(false, "Unsupported");
    } break;
    }
}

bool ShaderPrinter::TranslateTexelFetch(const Shader::FunctionCall* func_call)
{
    auto& os = m_Printer.stream();
    auto func_call_name = func_call->getNodeName();
    auto* args = func_call->getArguments();
    // First, move the texture name to the front.
    auto iter = args->current();
    TGE_ASSERT(iter != args->end(), "Expected at least one argument");
    auto iter_sampler = m_SamplerAssoc.find(iter->getNodeName());
    if(iter_sampler == m_SamplerAssoc.end())
    {
        Log(LogLevel::Error, "Expecting valid sampler associated with texture: ", iter->getNodeName());
        return false;
    }
    iter->accept(this);
    ++iter;
    
    // Next, we modify the texture coordinates, so that they match the weird HLSL format.
    auto* func_decl = func_call->getFunction();
    auto* expr_list = func_decl->getArgumentExpressions();
    auto sampler_expr = expr_list->current();
    auto texcoord_expr = sampler_expr+1;
    auto* texcoord_type = texcoord_expr->extract<Shader::Declaration>()->getVariables()->extract<Shader::Variable>()->getType();
    if(texcoord_type->getTypeEnum() == Shader::ElementType::Scalar)
    {
        if(sampler_expr->getNodeName() == "samplerBuffer"  ||
            sampler_expr->getNodeName() == "isamplerBuffer" ||
            sampler_expr->getNodeName() == "usamplerBuffer")
        {
            os << ".Load(" << iter_sampler->second << ", int(";
            iter->accept(this);
            os << ")";
        }
        else
        {
            os << ".Load(" << iter_sampler->second << ", int2(int(";
            iter->accept(this);
            os << "), ";
            ++iter;
            TGE_ASSERT(iter != args->end(), "Expected valid LOD.");
            iter->accept(this);
            os << ")";
        }
    }
    else if(texcoord_type->getTypeEnum() == Shader::ElementType::Vector)
    {
        auto* vector_type = texcoord_type->extract<Shader::VectorType>();
        auto sampler_name = sampler_expr->extract<Shader::Declaration>()->getVariables()->extract<Shader::Variable>()->getType()->getNodeName();
        if(sampler_name == "sampler2DMSArray"  || sampler_name == "sampler2DMS"  ||
            sampler_name == "isampler2DMSArray" || sampler_name == "isampler2DMS" ||
            sampler_name == "usampler2DMSArray" || sampler_name == "usampler2DMS")
        {
            os << ".Load(" << iter_sampler->second << ", int" << vector_type->getDimension() << "(";
            iter->accept(this);
            os << ")";
        }
        else
        {
            auto dim = vector_type->getDimension();
            os << ".Load(" << iter_sampler->second << ", int" << dim+1 << "(int" << dim << "(";
            iter->accept(this);
            os << "), ";
            ++iter;
            TGE_ASSERT(iter != args->end(), "Expected valid LOD.");
            iter->accept(this);
            os << ")";
        }
    }
    else
        TGE_ASSERT(false, "Unexpected sampling type");
    
    while(++iter != args->end())
    {
        os << ", ";
        iter->accept(this);
    }
    os << ")";
    return true;
}

ShaderPrinter::ShaderPrinter(Shader::Driver& driver, const char* filename, std::ostream& os, const string* opts, size_t opts_count, uint32 flags)
    :   m_Printer(os, flags),
        m_Driver(driver),
        m_Filename(filename),
        m_Options(opts),
        m_OptionCount(opts_count),
        m_Valid(true)
{
    m_FunctionTranslator["texelFetch"] = &ShaderPrinter::TranslateTexelFetch;
    m_FunctionTranslator["texelFetchOffset"] = &ShaderPrinter::TranslateTexelFetch;
/*    m_FunctionTranslator["texture"] // TODO
    m_FunctionTranslator["textureGather"]
    m_FunctionTranslator["textureGatherOffset"]
    m_FunctionTranslator["textureGatherOffsets"]
    m_FunctionTranslator["textureGrad"]
    m_FunctionTranslator["textureGradOffset"]
    m_FunctionTranslator["textureLod"]
    m_FunctionTranslator["textureLodOffset"]
    m_FunctionTranslator["textureOffset"]
    m_FunctionTranslator["textureProj"]
    m_FunctionTranslator["textureProjGrad"]
    m_FunctionTranslator["textureProjGradOffset"]
    m_FunctionTranslator["textureProjLod"]
    m_FunctionTranslator["textureProjLodOffset"]
    m_FunctionTranslator["textureProjOffset"]
    m_FunctionTranslator["textureQueryLevels"]
    m_FunctionTranslator["textureQueryLod"]
    m_FunctionTranslator["textureSize"]
    m_FunctionTranslator["imageAtomicAdd"]
    m_FunctionTranslator["imageAtomicAnd"]
    m_FunctionTranslator["imageAtomicCompSwap"]
    m_FunctionTranslator["imageAtomicExchange"]
    m_FunctionTranslator["imageAtomicMax"]
    m_FunctionTranslator["imageAtomicMin"]
    m_FunctionTranslator["imageAtomicOr"]
    m_FunctionTranslator["imageAtomicXor"]
    m_FunctionTranslator["imageLoad"]
    m_FunctionTranslator["imageSize"]
    m_FunctionTranslator["imageStore"]*/
}

ShaderPrinter::~ShaderPrinter()
{
}

void ShaderPrinter::visit(const Shader::VectorType* vector_type)
{
    m_Printer.stream() << vector_type->getBasicType()->getNodeName() << vector_type->getDimension();
}

void ShaderPrinter::visit(const Shader::MatrixType* matrix_type)
{
    m_Printer.stream() << matrix_type->getBasicType()->getNodeName() << matrix_type->getRows() << "x" << matrix_type->getColumns();
}

void ShaderPrinter::visit(const Shader::Variable* var)
{
    string var_name = var->getNodeName();
    if(var->getNodeName() == "tge_DrawID")
    {
        m_Printer.stream() << "shader_in__.InstanceID__";
        return;
    }

    for(auto i = m_InputSignature.begin(), iend = m_InputSignature.end(); i != iend; ++i)
        if(*i == var_name)
        {
            m_Printer.stream() << "shader_in__." << var_name;
            return;
        }
    for(auto i = m_OutputSignature.begin(), iend = m_OutputSignature.end(); i != iend; ++i)
        if(*i == var_name)
        {
            // That's safer because we are sure that it is vertex shader and no one is screwing with us.
            if(var_name == "gl_Position")
                m_Printer.stream() << "shader_out__.Position__";
            else
                m_Printer.stream() << "shader_out__." << var_name;
            return;
        }
    Shader::PrintNode(&m_Printer, var);
}



void ShaderPrinter::visit(const Shader::FunctionCall* func_call)
{
    auto func_call_name = func_call->getNodeName();
    auto iter = m_FunctionTranslator.find(func_call_name);

    if(iter == m_FunctionTranslator.end())
    {
        Shader::PrintNode(this, &m_Printer, func_call);
    }
    else
    {
        ((*this).*(iter->second))(func_call);
    }
}

void ShaderPrinter::visit(const Shader::InvariantDeclaration* invar_decl)
{
    // Unsupported in HLSL
}

class Generator: public Shader::VisitorInterface
{
    Shader::Driver&            m_Driver;
    std::stringstream          m_RawImportStream;
    ShaderPrinter              m_RawImport;
    size_t                     m_StructBufferBinding = 0;

    const Shader::OptionsDeclaration* m_OptionsDeclaration = nullptr;

    bool                       m_Valid;
    Shader::EffectDescription& m_Effect;
    FileLoader*                m_FileLoader;
    const string*              m_Options;
    size_t                     m_OptionCount;
public:
    Generator(Shader::Driver& driver, const string* opts, size_t count, Shader::EffectDescription& effect, const char* filename, FileLoader* include_loader);
    virtual ~Generator();

    virtual void visit(const Location& loc) final { m_RawImport.visit(loc); }
    virtual void visit(const AST::Value<float>* value) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const AST::Value<int>* value) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const AST::Value<unsigned>* value) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const AST::Value<bool>* value) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const AST::Value<string>* value) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const AST::ListElement* lst) final;
    virtual void visit(const AST::Block* _block) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const AST::StringLiteral* value) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Typedef* _typedef) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Parentheses* parentheses) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::FunctionCall* func_call) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::ConstructorCall* constructor) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::ScalarType* scalar_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::VectorType* vector_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::MatrixType* matrix_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::SamplerType* sampler_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::ArrayType* array_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Variable* var) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Declaration* decl) final;
    virtual void visit(const Shader::MemberVariable* mem_var) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::ArrayElementVariable* array_elem) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::InvariantDeclaration* invar_decl) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::BinaryOperator* binop) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::UnaryOperator* unaryop) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::TernaryIf* ternary_if) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::WhileStatement* while_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::ForStatement* for_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::SwitchStatement* switch_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::CaseStatement* case_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::JumpStatement* jump_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::ReturnStatement* return_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Import* _import) final;
    virtual void visit(const Shader::ShaderDeclaration* _shader) final;
    virtual void visit(const Shader::OptionsDeclaration* _opt_decl) final;
    virtual void visit(const Shader::Optional* _opt) final  { PrintOptional(this, m_RawImport.getPrinter(), _opt, m_Options, m_OptionCount); }
    virtual void visit(const Shader::Option* _opt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::StructType* _struct) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::FunctionDefinition* func_def) final;
    virtual void visit(const Shader::FunctionDeclaration* func_decl) final;
    virtual void visit(const Shader::IfStatement* if_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Type* type_stmt) final;
    virtual void visit(const Shader::Buffer* buffer) final;
    // Some types that should not appear in AST
    virtual void visit(const Shader::FunctionSet*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::Expression*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::IntermFuncNode*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::FuncDeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::DeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::VarDeclList*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::Value<Shader::ShaderType>*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }

    bool isValid() const { return m_Valid; }
};

Generator::Generator(Shader::Driver& driver, const string* opts, size_t opts_count, Shader::EffectDescription& effect, const char* filename, FileLoader* include_loader)
    :   m_Driver(driver),
        m_Effect(effect),
        m_RawImport(driver, filename, m_RawImportStream, opts, opts_count, AST::TGE_AST_PRINT_LINE_LOCATION),
        m_Valid(true),
        m_FileLoader(include_loader),
        m_Options(opts),
        m_OptionCount(opts_count) {}

Generator::~Generator() {}

string ConvertHLSLVersion(Shader::ShaderType _type)
{
    string shader_type;
    switch(_type)
    {
    case Shader::ShaderType::VertexShader: shader_type = "vs_"; break;
    case Shader::ShaderType::TessellationControlShader: shader_type = "hs_"; break;
    case Shader::ShaderType::TessellationEvaluationShader: shader_type = "ds_"; break;
    case Shader::ShaderType::GeometryShader: shader_type = "gs_"; break;
    case Shader::ShaderType::FragmentShader: shader_type = "ps_"; break;
    case Shader::ShaderType::ComputeShader: shader_type = "cs_"; break;
    default:
        Log(LogLevel::Error, "Unknown shader type: ", (int)_type);
        return "";
    }
    
    return shader_type + "5_0";
}

void Generator::visit(const Shader::Buffer* _buffer)
{
    m_RawImport.visit(_buffer);

    ConvertBuffer(m_Options, m_OptionCount, 0, _buffer, &m_Effect);
}

void Generator::visit(const Shader::Declaration* decl)
{
    if(decl == nullptr)
        return;
    auto* var_node = decl->getVariables();
    auto var_type = var_node->getNodeType();
    TGE_ASSERT(var_type == Shader::TGE_EFFECT_VARIABLE ||
               var_type == Shader::TGE_EFFECT_TYPE, "Expecting variable or type declaration");
    if(var_type == Shader::TGE_EFFECT_TYPE)
    {
        m_RawImport.visit(decl);
    }
    else if(var_type == Shader::TGE_EFFECT_VARIABLE)
    {
        auto* var = var_node->extract<Shader::Variable>();
        if(var->getStorage() == Shader::StorageQualifier::StructBuffer)
        {
            auto name = var->getNodeName();
            auto* var_type = var->getType();
            TGE_ASSERT(var_type->getTypeEnum() == Shader::ElementType::Array, "Unexpected type");
            auto* array_type = var_type->extract<Shader::ArrayType>();
            auto* elem_type = array_type->getArrayElementType();
            Shader::ElementType type_enum = elem_type->getTypeEnum();
            TGE_ASSERT(type_enum == Shader::ElementType::Struct, "Structured buffer should have struct type");

            m_RawImportStream << "StructuredBuffer<" << elem_type->getNodeName() << "> " << name << ": register(t" << m_StructBufferBinding++ << ")";

            ConvertStructBuffer(m_Options, m_OptionCount, 0, var, &m_Effect);
        }
        else
        {
            Log(LogLevel::Error, "Unexpected variable declaration ", var->getNodeName(), ".");
            m_Valid = false;
            return;
        }
    }
}

void Generator::visit(const AST::ListElement* lst)
{
    for(auto i = lst->current(), iend = lst->end(); i != iend; ++i)
    {
        i->accept(this);
        if(!i->isBlockStatement())
            m_RawImport.stream() << ";\n";
    }
}

void Generator::visit(const Shader::Type* type_stmt)
{
    TGE_ASSERT(type_stmt->getTypeEnum() == Shader::ElementType::Struct, "Unexpected top level type declaration");
    type_stmt->accept(this);
}

// Same as GLFX
void Generator::visit(const Shader::Import* _import)
{
    // Imported data generally speaking is an external file that does not
    // get parsed. It is also possible to place some definitions that would
    // be automatically prepended before this block and undefined afterwards
    // to protect from polluting the global space.
    string name = _import->getNodeName();
    TGE_ASSERT(name[0] == '"' && name.back() == '"', "unexpected import layout");
    name.erase(name.begin());
    name.erase(name.end()-1);
    auto file_desc = CREATE_SCOPED_FILE(m_FileLoader, name);
    m_Effect.addImportedFile(name);
    auto& raw_import_stream = m_RawImport.stream();
    raw_import_stream << "#line 1 " << m_Effect.getImportedFileCount() << "\n";
    auto* def_list = _import->getBody();
    for(auto j = def_list->current(); j != def_list->end(); ++j)
    {
        TGE_ASSERT(j->getNodeType() == Shader::TGE_EFFECT_BINARY_OPERATOR, "Binary operator expected as part of import statement definitions list");
        auto* binop = j->extract<Shader::BinaryOperator>();
        TGE_ASSERT(binop->getOperation() == Shader::TGE_EFFECT_ASSIGN, "Assignment operator expected as part of import statement definitions list");
        raw_import_stream << "#define ";
        binop->getLHSOperand()->accept(&m_RawImport);
        raw_import_stream << " ";
        binop->getRHSOperand()->accept(&m_RawImport);
        raw_import_stream << "\n";
    }
    raw_import_stream.write(file_desc->Content, file_desc->ContentSize);
    for(auto j = def_list->current(); j != def_list->end(); ++j)
    {
        auto* binop = j->extract<Shader::BinaryOperator>();
        raw_import_stream << "#undef ";
        binop->getLHSOperand()->accept(&m_RawImport);
        raw_import_stream << "\n";
    }
}

void Generator::visit(const Shader::ShaderDeclaration* _shader)
{
    std::unique_ptr<Shader::ShaderDescription> shader_desc(new Shader::ShaderDescription);

    ShaderSignature             input_signature,
                                output_signature,
                                sampler_state;

    std::stringstream           in_signature_ss,
                                out_signature_ss,
                                ss;
                                
    // The actual trick out here is that we want the common stuff to be printed through
    // the extended version which accumulates all associations.
    ShaderPrinter common_printer(m_Driver, m_RawImport.getFilename(), ss, m_Options, m_OptionCount, AST::TGE_AST_PRINT_LINE_LOCATION),
                  in_printer(m_Driver, m_RawImport.getFilename(), in_signature_ss, m_Options, m_OptionCount, AST::TGE_AST_PRINT_LINE_LOCATION),
                  out_printer(m_Driver, m_RawImport.getFilename(), out_signature_ss, m_Options, m_OptionCount, AST::TGE_AST_PRINT_LINE_LOCATION);
    const Shader::FunctionDefinition* entrypoint = nullptr;
    
    auto shader_type = _shader->getType();
    if(shader_type == Shader::ShaderType::VertexShader)
    {
        output_signature.push_back("gl_Position");
        out_signature_ss << "struct shader_output_signature__\n"
                            "{\n"
                            "\tfloat4 Position__: SV_POSITION;\n";
        input_signature.push_back("InstanceID__");
        in_signature_ss << "struct shader_input_signature__\n"
                           "{\n";
    }
    size_t target = 0;
    for(auto j = _shader->getBody()->current(); j != _shader->getBody()->end(); ++j)
    {
        TGE_ASSERT(*j, "Valid node expected. Bad parsing beforehand.");
        if(j->getNodeType() == Shader::TGE_EFFECT_INVARIANT_DECLARATION)
            continue;
        if(j->getNodeType() == Shader::TGE_EFFECT_FUNCTION_DEFINITION && j->getNodeName() == "main")
        {
            entrypoint = j->extract<Shader::FunctionDefinition>();
        }
        else if(j->getNodeType() == Shader::TGE_EFFECT_DECLARATION)
        {
            auto* var_node = j->extract<Shader::Declaration>()->getVariables();
            if(var_node->getNodeType() != Shader::TGE_EFFECT_VARIABLE)
            {
                Log(LogLevel::Error, "Unsupported global declaration type");
                m_Valid = false;
                return;
            }
            auto* var = var_node->extract<Shader::Variable>();
            auto* layout = var->getLayout();
            
            Shader::VertexAttributeDescription vert_attr;
            bool vb_attrs = false;

            string prefix, suffix;
            if(layout)
            {
                for(auto k = layout->current(); k != layout->end(); ++k)
                {
                    auto* binop = k->extract<Shader::BinaryOperator>();
                    auto layout_id = binop->getLHSOperand()->extract<Shader::Identifier>()->getValue();
                    if(layout_id == "vb_offset")
                    {
                        vert_attr.Offset = binop->getRHSOperand()->extract<Shader::Value<int>>()->getValue();
                        vb_attrs = true;
                    }
                    else if(layout_id == "vb_format")
                    {
                        vert_attr.Format = TranslateDataFormat(binop->getRHSOperand()->extract<Shader::Value<string>>()->getValue());
                        vb_attrs = true;
                    }
                    else
                    {
                        Log(LogLevel::Error, k->getDeclarationLocation(), ": Unsupported layout qualifier: ", layout_id);
                        m_Valid = false;
                    }
                }
            }

            auto* var_type = var->getType();
            auto storage = var->getStorage();

            if(vb_attrs && storage != Shader::StorageQualifier::In)
            {
                Log(LogLevel::Error, var_node->getDeclarationLocation(), ": Vertex buffer format parameters are not supported for variables that are not of input type.");
                m_Valid = false;
                return;
            }

            std::stringstream* current_ss = &ss;
            ShaderPrinter* current_printer = &common_printer;
            switch(storage)
            {
            case Shader::StorageQualifier::StructBuffer:
            {
                auto name = var->getNodeName();
                auto* var_type = var->getType();
                TGE_ASSERT(var_type->getTypeEnum() == Shader::ElementType::Array, "Unexpected type");
                auto* array_type = var_type->extract<Shader::ArrayType>();
                auto* elem_type = array_type->getArrayElementType();
                TGE_ASSERT(elem_type->getTypeEnum() != Shader::ElementType::Struct, "Structured buffer should have struct type");

                common_printer.stream() << "StructuredBuffer<" << elem_type->getNodeName() << "> " << name << ";";
            } break;
            case Shader::StorageQualifier::Default: common_printer.visit(j->getDeclarationLocation()); break;
            case Shader::StorageQualifier::Const: common_printer.visit(j->getDeclarationLocation()); prefix = "const "; break;
            case Shader::StorageQualifier::In:
            {
                if(input_signature.empty())
                    in_signature_ss << "struct shader_input_signature__\n{" << std::endl;
                current_ss = &in_signature_ss;
                in_printer.visit(j->getDeclarationLocation());
                in_signature_ss << "\t";
                current_printer = &in_printer;
                input_signature.push_back(var->getNodeName());
                
                if(shader_type == Shader::ShaderType::VertexShader)
                {
                    if(vert_attr.Format == DataFormat::Unknown)
                    {
                        Log(LogLevel::Error, var_node->getDeclarationLocation(), ": Valid format must be specified for input attribute: ", var->getNodeName());
                        m_Valid = false;
                        return;
                    }
                    vert_attr.Name = var->getNodeName();
                    size_t dims;
                    switch(var_type->getTypeEnum())
                    {
                    case Shader::ElementType::Scalar: dims = 1; break;
                    case Shader::ElementType::Vector: dims = var_type->extract<Shader::VectorType>()->getDimension(); break;
                    default:
                    {
                        Log(LogLevel::Error, "Unsupported input attribute type");
                        m_Valid = false;
                        return;
                    } break;
                    }

                    if(DataFormatChannels(vert_attr.Format) != dims)
                    {
                        Log(LogLevel::Error, var_node->getDeclarationLocation(), ": Incompatible data format specified for input attribute: ", var->getNodeName());
                        m_Valid = false;
                        return;
                    }

                    m_Effect.addVertexAttribute(vert_attr);
                }
                else if(vb_attrs)
                {
                    Log(LogLevel::Error, var_node->getDeclarationLocation(), ": Input layout is only supported in vertex shader stage.");
                    m_Valid = false;
                }

            } break;
            case Shader::StorageQualifier::CentroidIn:
            {
                if(input_signature.empty())
                    in_signature_ss << "struct shader_input_signature__\n{" << std::endl;
                prefix = "centroid ";
                current_ss = &in_signature_ss;
                in_printer.visit(j->getDeclarationLocation());
                in_signature_ss << "\t";
                current_printer = &in_printer;
                input_signature.push_back(var->getNodeName());
            } break;
            case Shader::StorageQualifier::SampleIn: TGE_ASSERT(false, "Unsupported"); break;
            case Shader::StorageQualifier::Out:
            {
                if(output_signature.empty())
                    out_signature_ss << "struct shader_output_signature__\n{" << std::endl;
                current_ss = &out_signature_ss;
                out_printer.visit(j->getDeclarationLocation());
                out_signature_ss << "\t";
                current_printer = &out_printer;
                output_signature.push_back(var->getNodeName());
            } break;
            case Shader::StorageQualifier::CentroidOut:
            {
                if(output_signature.empty())
                    out_signature_ss << "struct shader_output_signature__\n{" << std::endl;
                prefix = "centroid ";
                current_ss = &out_signature_ss;
                out_printer.visit(j->getDeclarationLocation());
                out_signature_ss<< "\t";
                current_printer = &out_printer;
                output_signature.push_back(var->getNodeName());
            } break;
            case Shader::StorageQualifier::SampleOut: TGE_ASSERT(false, "Unsupported"); break;
            case Shader::StorageQualifier::InOut: TGE_ASSERT(false, "Unexpected storage format."); break;
            default:
                TGE_ASSERT(false, "Unsupported storage type.");
            }
            
            if(!prefix.empty())
                *current_ss << prefix << " ";
            var->getType()->accept(current_printer);
            *current_ss << " " << var->getNodeName() << suffix;
            if(shader_type == Shader::ShaderType::FragmentShader && current_ss == &out_signature_ss)
            {
                *current_ss << ": SV_TARGET" << target++;
            }
            else
            {
                *current_ss << ": PARAM_" << var->getNodeName();
            }
            *current_ss << ";\n";
        }
        else
        {
            common_printer.visit(j->getDeclarationLocation());
            j->accept(&common_printer);
            if(!j->isBlockStatement())
                ss << ";\n";
        }
    }

    if(!input_signature.empty())
    {
        in_signature_ss << "\tuint InstanceID__: SV_INSTANCEID;\n"
                           "};" << std::endl;
    }
    if(!output_signature.empty())
        out_signature_ss << "};" << std::endl;

    if(entrypoint == nullptr)
    {
        Log(LogLevel::Error, "Shader must contain valid entrypoint: ", ConvertShaderTypeToText(shader_type));
        m_Valid = false;
        return;
    }
    
    for(auto i = sampler_state.begin(), iend = sampler_state.end(); i != iend; ++i)
        ss << "SamplerState " << *i << ";\n";
    ss << m_RawImportStream.str() << in_signature_ss.str() << out_signature_ss.str();
    
    if(output_signature.empty())
        ss << "void ";
    else
        ss << "shader_output_signature__ ";
    
    ss << "TempestShaderMain(";
    if(!input_signature.empty())
        ss << "shader_input_signature__ shader_in__";
    // TODO: Add any stuff that are needed by different shading stages.
    ss << ")\n"
          "{\n"
          "\tshader_output_signature__ shader_out__;\n";
    
    common_printer.setInputSignature(input_signature);
    common_printer.setOutputSignature(output_signature);
        
    common_printer.setIndentation(1);
        
    common_printer.visit(entrypoint->getBody());
    
    ss << "\treturn shader_out__;\n"
       << "}\n";
    
    shader_desc->setAdditionalOptions(ConvertHLSLVersion(shader_type));

    shader_desc->appendContent(ss.str());
    if(!m_Effect.trySetShader(shader_type, shader_desc.release()))
    {
        Log(LogLevel::Error, "Duplicate shader types: ", ConvertShaderTypeToText(shader_type), "\n"
                             "Try using \"options\" to eliminate some of the shaders.");
        m_Valid = false;
        return;
    }
    
    if(!common_printer.isValid() ||
       !in_printer.isValid() ||
       !out_printer.isValid())
    {
        m_Valid = false;
        return;
    }
}

void Generator::visit(const Shader::OptionsDeclaration* _opt_decl)
{
    if(m_OptionsDeclaration)
    {
        Log(LogLevel::Error, "More than a single options declaration is unsupported");
        m_Valid = false;
        return;
    }
    m_OptionsDeclaration = _opt_decl;
    size_t decl_opts_count = m_OptionsDeclaration->getOptionCount();
    for(size_t i = 0, iend = m_OptionCount; i < iend; ++i)
    {
        auto& opt = m_Options[i];
        if(m_OptionsDeclaration->getOptionIndex(opt) == decl_opts_count)
        {
            Log(LogLevel::Error, "Unspecified option ", opt);
            m_Valid = false;
        }
    }
}

void Generator::visit(const Shader::FunctionDefinition* func_def)
{
    m_RawImport.visit(func_def);
}

void Generator::visit(const Shader::FunctionDeclaration* func_decl)
{
    m_RawImport.visit(func_decl);
    m_RawImportStream << ";\n";
}

bool LoadEffect(const string& filename, FileLoader* loader, const string* opts, size_t opts_count, uint32 flags, Shader::EffectDescription& effect)
{
    Shader::Driver  effect_driver;
    auto            parse_ret = effect_driver.parseFile(filename);
    if(!parse_ret)
    {
        std::stringstream ss;
        ss << "The application has failed to parse a shader program (refer to the error log for more information): " << filename << std::endl;
        Log(LogLevel::Error, ss.str());
        TGE_ASSERT(parse_ret, ss.str());
        return false;
    }

    AST::List*    root_node = effect_driver.getASTRoot()->extract<AST::List>();
    if(!root_node)
        return false;

    DXFX::Generator _generator(effect_driver, opts, opts_count, effect, filename.c_str(), loader);
    _generator.visit(root_node);
    return _generator.isValid();
}
}
}
