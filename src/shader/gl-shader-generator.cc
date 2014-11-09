/*   The MIT License
 *
 *   Tempest Engine
 *   Copyright (c) 2013 Zdravko Velinov
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

#include "tempest/shader/gl-shader-generator.hh"
#include "tempest/shader/shader-ast.hh"
#include "tempest/shader/shader-driver.hh"
#include "tempest/parser/file-loader.hh"
#include "tempest/utils/patterns.hh"

#include <unordered_map>

namespace Tempest
{
namespace GLFX
{
class ShaderPrinter;

void PrintBuffer(AST::VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Shader::Buffer* buffer, size_t* binding_counter);

class ShaderPrinter: public Shader::VisitorInterface
{
    // Because we have scope, we must initialize a new variables map
    AST::PrinterInfrastructure m_Printer;
    
    Shader::ShaderType         m_ShaderType = Shader::ShaderType::GenericShader;
    bool                       m_DrawIDInUse = false;
    
    size_t                     m_BindingCounter;
    
public:
    ShaderPrinter(std::ostream& os, size_t binding_counter, uint32 flags)
        :   m_Printer(os, flags),
            m_BindingCounter(binding_counter) {}
    virtual ~ShaderPrinter()=default;

    std::ostream& stream() { return m_Printer.stream(); }
    size_t getIndentation() const { return m_Printer.getIndentation(); }

    void setShaderType(Shader::ShaderType shader_type) { m_ShaderType = shader_type; }
    bool isDrawIDInUse() const { return m_DrawIDInUse; }
    
    // That's pretty much top-level declaration, so no recursion is supported
    void printSubroutine(const Shader::Subroutine* subroutine) { Shader::PrintNode(this, &m_Printer, subroutine); }
    
    virtual void visit(const Location& loc) final { AST::PrintLocation(&m_Printer, loc); }
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
    virtual void visit(const Shader::FunctionCall* func_call) final { Shader::PrintNode(this, &m_Printer, func_call); }
    virtual void visit(const Shader::SubroutineCall* subroutine_call) final { Shader::PrintNode(this, &m_Printer, subroutine_call); }
    virtual void visit(const Shader::ConstructorCall* constructor) final { Shader::PrintNode(this, &m_Printer, constructor); }
    virtual void visit(const Shader::ScalarType* scalar_type) final { Shader::PrintNode(&m_Printer, scalar_type); }
    virtual void visit(const Shader::VectorType* vector_type) final { Shader::PrintNode(&m_Printer, vector_type); }
    virtual void visit(const Shader::MatrixType* matrix_type) final { Shader::PrintNode(&m_Printer, matrix_type); }
    virtual void visit(const Shader::SamplerType* sampler_type) final { Shader::PrintNode(&m_Printer, sampler_type); }
    virtual void visit(const Shader::ArrayType* array_type) final { Shader::PrintNode(this, &m_Printer, array_type); }
    virtual void visit(const Shader::StructType* _struct) final { Shader::PrintNode(this, &m_Printer, _struct); }
    virtual void visit(const Shader::Variable* var) final;
    virtual void visit(const Shader::Declaration* decl) final;
    virtual void visit(const Shader::MemberVariable* mem_var) final { Shader::PrintNode(this, &m_Printer, mem_var); }
    virtual void visit(const Shader::ArrayElementVariable* array_elem) final { Shader::PrintNode(this, &m_Printer, array_elem); }
    virtual void visit(const Shader::InvariantDeclaration* invar_decl) final { Shader::PrintNode(this, &m_Printer, invar_decl); }
    virtual void visit(const Shader::BinaryOperator* binop) final { Shader::PrintNode(this, &m_Printer, binop); }
    virtual void visit(const Shader::UnaryOperator* unaryop) final { Shader::PrintNode(this, &m_Printer, unaryop); }
    virtual void visit(const Shader::TernaryIf* ternary_if) final { Shader::PrintNode(this, &m_Printer, ternary_if); }
    virtual void visit(const Shader::WhileStatement* while_stmt) final { Shader::PrintNode(this, &m_Printer, while_stmt); }
    virtual void visit(const Shader::ForStatement* for_stmt) final { Shader::PrintNode(this, &m_Printer, for_stmt); }
    virtual void visit(const Shader::SwitchStatement* switch_stmt) final { Shader::PrintNode(this, &m_Printer, switch_stmt); }
    virtual void visit(const Shader::CaseStatement* case_stmt) final { Shader::PrintNode(this, &m_Printer, case_stmt); }
    virtual void visit(const Shader::JumpStatement* jump_stmt) final { Shader::PrintNode(&m_Printer, jump_stmt); }
    virtual void visit(const Shader::ReturnStatement* return_stmt) final { Shader::PrintNode(this, &m_Printer, return_stmt); }
    virtual void visit(const Shader::Profile* _profile) final { Shader::PrintNode(&m_Printer, _profile); }
    virtual void visit(const Shader::Technique* _technique) final { _technique->printList(this, &m_Printer, "technique"); }
    virtual void visit(const Shader::Import* _import) final { _import->printList(this, &m_Printer, "import"); }
    virtual void visit(const Shader::ShaderDeclaration* _shader) final { Shader::PrintNode(this, &m_Printer, _shader); }
    virtual void visit(const Shader::CompiledShader* compiled_shader) final { Shader::PrintNode(&m_Printer, compiled_shader); }
    virtual void visit(const Shader::Pass* _pass) final { _pass->printList(this, &m_Printer, "pass"); }
    virtual void visit(const Shader::IfStatement* if_stmt) final { Shader::PrintNode(this, &m_Printer, if_stmt); }
    virtual void visit(const Shader::Type* type_stmt) final { Shader::PrintNode(this, type_stmt); }
    virtual void visit(const Shader::Buffer* buffer) final { PrintBuffer(this, &m_Printer, buffer, &m_BindingCounter); }
    virtual void visit(const Shader::Subroutine* subroutine) final { Shader::PrintNode(this, &m_Printer, subroutine); }
    // Some types that should not appear in AST
    virtual void visit(const Shader::IntermFuncNode*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FunctionSet*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Expression*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FuncDeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::DeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::VarDeclList*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const AST::Value<Shader::ShaderType>*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
};

void ShaderPrinter::visit(const Shader::Declaration* decl)
{
    auto* vars = decl->getVariables();
    if(vars->getNodeType() == Shader::TGE_EFFECT_VARIABLE)
    {
        auto* var = vars->extract<Shader::Variable>();
        if(var->getStorage() == Shader::StorageQualifier::StructBuffer)
        {
            auto& os = m_Printer.stream();
            for(size_t i = 0, indentation = m_Printer.getIndentation(); i < indentation; ++i)
                os << "\t";
            os << "layout(std430, binding = " << m_BindingCounter++ << ") buffer " << var->getNodeName() << "_StructBuffer "
                << "{\n";
            for(size_t i = 0, indentation = m_Printer.getIndentation(); i < indentation + 1; ++i)
                os << "\t";
            os << var->getType()->getNodeName() << " " << var->getNodeName() << "[]";
            for(size_t i = 0, indentation = m_Printer.getIndentation(); i < indentation; ++i)
                os << "\t";
            os << "};\n";
            return;
        }
    }
    
    Shader::PrintNode(this, &m_Printer, decl);
}

void ShaderPrinter::visit(const Shader::Variable* var)
{
    if(var->getNodeName() == "tge_DrawID")
    {
        m_Printer.stream() << "gl_DrawIDARB";
    }
    else
    {
        Shader::PrintNode(&m_Printer, var);
    }
}

class Generator: public Shader::VisitorInterface
{
    std::stringstream          m_RawImportStream;
    ShaderPrinter              m_RawImport;

    size_t                     m_BindingCounter;
    bool                       m_Valid;
    Shader::EffectDescription  m_Effect;
    FileLoader*                m_FileLoader;
public:
    Generator(FileLoader* include_loader);
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
    virtual void visit(const Shader::SubroutineCall* subroutine_call) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::ConstructorCall* constructor) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::ScalarType* scalar_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::VectorType* vector_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::MatrixType* matrix_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::SamplerType* sampler_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::ArrayType* array_type) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::StructType* _struct) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Variable* var) final  { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
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
    virtual void visit(const Shader::Profile* _profile) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Technique* _technique) final;
    virtual void visit(const Shader::Import* _import) final;
    virtual void visit(const Shader::ShaderDeclaration* _shader) final;
    virtual void visit(const Shader::FunctionDefinition* func_def) final;
    virtual void visit(const Shader::FunctionDeclaration* func_decl) final;
    virtual void visit(const Shader::CompiledShader* compiled_shader) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Pass* _pass) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::IfStatement* if_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Type* type_stmt) final;
    virtual void visit(const Shader::Buffer* buffer) final;
    virtual void visit(const Shader::Subroutine* subroutine) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    // Some types that should not appear in AST
    virtual void visit(const Shader::IntermFuncNode*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FunctionSet*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::Expression*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::FuncDeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::DeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::VarDeclList*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::Value<Shader::ShaderType>*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }

    bool isValid() const { return m_Valid; }
    
    Shader::EffectDescription getEffect() const { return m_Effect; }
};

Generator::Generator(FileLoader* include_loader)
    :   m_RawImport(m_RawImportStream, m_BindingCounter, AST::TGE_AST_PRINT_LINE_LOCATION),
        m_Valid(true),
        m_FileLoader(include_loader) {}

Generator::~Generator() {}

string TranslateGLSLVersion(const string& profile_name)
{
    // I prefer this over an lazy hack that copies numbers because it is much safer.
    if(profile_name == "glsl_1_4_0")
        return "#version 140\n\n";
    else if(profile_name == "glsl_1_5_0")
        return "#version 150\n\n";
    else if(profile_name == "glsl_3_3_0")
        return "#version 330\n\n";
    else if(profile_name == "glsl_4_0_0")
        return "#version 400\n\n";
    else if(profile_name == "glsl_4_1_0")
        return "#version 410\n\n";
    else if(profile_name == "glsl_4_2_0")
        return "#version 420\n\n";
    else if(profile_name == "glsl_4_4_0")
        return "#version 440\n"
               "#extension GL_ARB_shader_draw_parameters : require\n"
               "#extension GL_ARB_bindless_texture : require\n\n";
    
    Log(LogLevel::Error, "Unknown profile: ", profile_name);
    return "";
}

void PrintBuffer(AST::VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Shader::Buffer* buffer, size_t* binding_counter)
{
    switch(buffer->getBufferType())
    {
    default: TGE_ASSERT(false, "Unknown buffer type");
    case Shader::BufferType::Constant:
    {
        for(size_t i = 0, indentation = printer->getIndentation(); i < indentation; ++i)
            printer->stream() << "\t";
        printer->stream() << "layout(std140, binding = " << (*binding_counter)++ << ") uniform " << buffer->getNodeName()
                          << "{\n";
        auto* list = buffer->getBody();
        visitor->visit(list);
        printer->stream() << "};\n";
    } break;
    case Shader::BufferType::Regular:
    {
        for(size_t i = 0, indentation = printer->getIndentation(); i < indentation; ++i)
            printer->stream() << "\t";
        printer->stream() << "layout(std430, binding = " << (*binding_counter)++ << ") buffer " << buffer->getNodeName()
                          << "{\n";
        auto* list = buffer->getBody();
        visitor->visit(list);
        printer->stream() << "};\n";
    } break;
    }
}

static void ConvertVariable(const string& base, const Shader::Variable* var, size_t* offset, Shader::BufferDescription* buf_desc);

static void ConvertType(const string& base, const Shader::Type* _type, UniformValueType* uniform_type, size_t* elem_size, size_t* offset, Shader::BufferDescription* buf_desc)
{
    switch(_type->getTypeEnum())
    {
    case Shader::ElementType::Vector:
    {
        auto* vector_type = _type->extract<Shader::VectorType>();
        auto* basic_type = vector_type->getBasicType()->extract<Shader::ScalarType>();
        switch(vector_type->getDimension())
        {
        case 2: 
        {
            if(basic_type->getNodeName() == "int")
                *uniform_type = UniformValueType::IntegerVector2;
            else if(basic_type->getNodeName() == "bool")
                *uniform_type = UniformValueType::BooleanVector2;
            else if(basic_type->getNodeName() == "uint")
                *uniform_type = UniformValueType::UnsignedIntegerVector2;
            else if(basic_type->getNodeName() == "float")
                *uniform_type = UniformValueType::Vector2;
            else
                TGE_ASSERT(false, "Unknown vector type");
        } break;
        case 3:
        {
            if(basic_type->getNodeName() == "int")
                *uniform_type = UniformValueType::IntegerVector3;
            else if(basic_type->getNodeName() == "bool")
                *uniform_type = UniformValueType::BooleanVector3;
            else if(basic_type->getNodeName() == "uint")
                *uniform_type = UniformValueType::UnsignedIntegerVector3;
            else if(basic_type->getNodeName() == "float")
                *uniform_type = UniformValueType::Vector3;
            else
                TGE_ASSERT(false, "Unknown vector type");
        } break;
        case 4:
        {
            if(basic_type->getNodeName() == "int")
                *uniform_type = UniformValueType::IntegerVector4;
            else if(basic_type->getNodeName() == "bool")
                *uniform_type = UniformValueType::BooleanVector4;
            else if(basic_type->getNodeName() == "uint")
                *uniform_type = UniformValueType::UnsignedIntegerVector4;
            else if(basic_type->getNodeName() == "float")
                *uniform_type = UniformValueType::Vector4;
            else
                TGE_ASSERT(false, "Unknown vector type");
        } break;
        default:
            TGE_ASSERT(false, "Unknown vector type");
        }
        *elem_size = UniformValueTypeSize(*uniform_type);
    } break;
    case Shader::ElementType::Matrix:
    {
        auto* matrix_type = _type->extract<Shader::MatrixType>();
        size_t rows = matrix_type->getRows();
        size_t columns = matrix_type->getColumns();
        switch(columns)
        {
        case 2:
            switch(rows)
            {
            case 2: *uniform_type = UniformValueType::Matrix2; break;
            case 3: *uniform_type = UniformValueType::Matrix2x3; break;
            case 4: *uniform_type = UniformValueType::Matrix2x4; break;
            default: TGE_ASSERT(false, "Unsupported matrix type"); break;
            }
            break;
        case 3:
            switch(rows)
            {
            case 2: *uniform_type = UniformValueType::Matrix3x2; break;
            case 3: *uniform_type = UniformValueType::Matrix3; break;
            case 4: *uniform_type = UniformValueType::Matrix3x4; break;
            default: TGE_ASSERT(false, "Unsupported matrix type"); break;
            }
            break;
        case 4:
            switch(rows)
            {
            case 2: *uniform_type = UniformValueType::Matrix4x2; break;
            case 3: *uniform_type = UniformValueType::Matrix4x3; break;
            case 4: *uniform_type = UniformValueType::Matrix4; break;
            default: TGE_ASSERT(false, "Unsupported matrix type"); break;
            }
            break;
        default: TGE_ASSERT(false, "Unsupported matrix type"); break;
        }
        *elem_size = UniformValueTypeSize(*uniform_type);
    } break;
    case Shader::ElementType::Struct:
    {
        *offset = (*offset + 4*sizeof(float) - 1) & ~(4*sizeof(float) - 1);
        size_t struct_offset = 0; // members are in relative offset units
        auto* struct_type = _type->extract<Shader::StructType>();
        auto* struct_body = struct_type->getBody();
        for(auto elem_iter = struct_body->current(), elem_iter_end = struct_body->end(); elem_iter != elem_iter_end; ++elem_iter)
        {
            TGE_ASSERT(elem_iter->getNodeType() == Shader::TGE_EFFECT_DECLARATION, "Expecting only declarations");
            auto* decl = elem_iter->extract<Shader::Declaration>();
            auto* vars = decl->getVariables();
            switch(vars->getNodeType())
            {
            case Shader::TGE_EFFECT_VARIABLE:
            {
                ConvertVariable(base, vars->extract<Shader::Variable>(), &struct_offset, buf_desc);
            } break;
            case Shader::TGE_AST_LIST_ELEMENT:
            {
                auto _list = vars->extract<AST::ListElement>();
                for(auto var_iter = _list->current(), var_iter_end = _list->end(); var_iter != var_iter_end; ++var_iter)
                {
                    TGE_ASSERT(var_iter->getNodeType() == Shader::TGE_EFFECT_VARIABLE, "Expecting variable");
                    auto* var = var_iter->extract<Shader::Variable>();
                    ConvertVariable(base, var, &struct_offset, buf_desc);    
                }
            } break;
            default: TGE_ASSERT(false, "Unexpected node");
            }
        }
        *uniform_type = UniformValueType::Struct;
        *elem_size = struct_offset;
    } break;
    default: TGE_ASSERT(false, "Unexpected type"); break;
    }
    TGE_ASSERT(*elem_size > 0, "Element size should be greater than one. Otherwise, it is pointless to define it");
}

static size_t GetAlignment(UniformValueType _type)
{
    switch(_type)
    {
    case UniformValueType::Float: return sizeof(float);
    case UniformValueType::Vector2: return 2*sizeof(float);
    case UniformValueType::Vector3: return 4*sizeof(float);
    case UniformValueType::Vector4: return 4*sizeof(float);
    case UniformValueType::Integer: return sizeof(int32);
    case UniformValueType::IntegerVector2: return 2*sizeof(int32);
    case UniformValueType::IntegerVector3: return 4*sizeof(int32);
    case UniformValueType::IntegerVector4: return 4*sizeof(int32);
    case UniformValueType::UnsignedInteger: return sizeof(uint32);
    case UniformValueType::UnsignedIntegerVector2: return 2*sizeof(uint32);
    case UniformValueType::UnsignedIntegerVector3: return 4*sizeof(uint32);
    case UniformValueType::UnsignedIntegerVector4: return 4*sizeof(uint32);
    case UniformValueType::Boolean: return sizeof(uint32);
    case UniformValueType::BooleanVector2: return 2*sizeof(uint32);
    case UniformValueType::BooleanVector3: return 4*sizeof(uint32);
    case UniformValueType::BooleanVector4: return 4*sizeof(uint32);
    case UniformValueType::Matrix2: return 2*2*sizeof(float);
    case UniformValueType::Matrix3: return 3*3*sizeof(float);
    case UniformValueType::Matrix4: return 4*4*sizeof(float);
    case UniformValueType::Matrix2x3: return 2*4*sizeof(float);
    case UniformValueType::Matrix2x4: return 2*4*sizeof(float);
    case UniformValueType::Matrix3x2: return 3*2*sizeof(float);
    case UniformValueType::Matrix3x4: return 3*4*sizeof(float);
    case UniformValueType::Matrix4x2: return 4*2*sizeof(float);
    case UniformValueType::Matrix4x3: return 4*4*sizeof(float);
    case UniformValueType::Texture: return sizeof(uint64);
    case UniformValueType::Struct: return 4*sizeof(float);
    case UniformValueType::SubroutineUniform: return sizeof(uint32);
    case UniformValueType::SubroutineFunction:
    default: TGE_ASSERT(false, "Unexpected uniform type");
    }
    return 0;
}

static void ConvertVariable(const string& base, const Shader::Variable* var, size_t* offset, Shader::BufferDescription* buf_desc)
{
    UniformValueType uniform_type;
    size_t           elem_size,
                     array_size = 1;
    auto*            _type = var->getType();
    auto             type_enum = _type->getTypeEnum();
    string           var_name = var->getNodeName();
    
    switch(type_enum)
    {
    case Shader::ElementType::Struct:
    case Shader::ElementType::Vector:
    case Shader::ElementType::Matrix:
    {
        ConvertType(base.empty() ? var_name : (base + "." + var_name), _type, &uniform_type, &elem_size, offset, buf_desc);
    } break;
    case Shader::ElementType::Array:
    {
        auto* array_type = _type->extract<Shader::ArrayType>();
        ConvertType(base.empty() ? var_name : (base + "." + var_name), array_type->getBasicType(), &uniform_type, &elem_size, offset, buf_desc);
        TGE_ASSERT(array_size == 1, "Arrays of arrays are unsupported");
        auto* size = array_type->getSize();
        if(*size)
        {
            array_size = size->extract<AST::Value<int>>()->getValue();
        }
        else
        {
            array_size = 0; // infinite
            buf_desc->setResizablePart(elem_size);
        }
        *offset = (*offset + 4*sizeof(float) - 1) & ~(4*sizeof(float) - 1);
    } break;
    case Shader::ElementType::Sampler:
    {
        auto* sampler_type = _type->extract<Shader::SamplerType>();
        uniform_type = UniformValueType::Texture;
        elem_size = sizeof(uint64);
        // TODO
        
    } break;
    default:
        TGE_ASSERT(false, "Unsupported type"); break;
    }
    
    auto alignment = GetAlignment(uniform_type);
    *offset = (*offset + alignment - 1) & ~(alignment - 1);

    buf_desc->addBufferElement(Shader::BufferElement(*offset, uniform_type, base.empty() ? var_name : (base + "." + var_name), elem_size, array_size));
    
    if(array_size > 1 || uniform_type == UniformValueType::Struct)
    {
        *offset += array_size*((elem_size + 4*sizeof(float) - 1) & ~(4*sizeof(float) - 1));
    }
    else
    {
        *offset += elem_size;
    }
}

static void ConvertSubroutine(const string& base, const Shader::Type* _type, UniformValueType* uniform_type, size_t* elem_size, size_t* array_size)
{
    *array_size = 1;
    auto type_enum = _type->getTypeEnum();
    switch(type_enum)
    {
    case Shader::ElementType::Array:
    {
        auto* array_type = _type->extract<Shader::ArrayType>();
        ConvertSubroutine(base, array_type->getBasicType(), uniform_type, elem_size, array_size);
        TGE_ASSERT(*array_size == 1, "Arrays of arrays are unsupported");
        auto* size = array_type->getSize();
        TGE_ASSERT(*size, "Variable size arrays are not supported");
        *array_size = size->extract<AST::Value<int>>()->getValue();
    } break;
    case Shader::ElementType::Subroutine:
    {
        *uniform_type = UniformValueType::SubroutineUniform;
        *elem_size = UniformValueTypeSize(*uniform_type);
    } break;
    default:
        TGE_ASSERT(false, "Unsupported type"); break;
    }
}

void Generator::visit(const Shader::Declaration* decl)
{
    if(decl == nullptr)
        return;
    auto* var_node = decl->getVariables();
    auto var_type = var_node->getNodeType();
    TGE_ASSERT(var_type == Shader::TGE_EFFECT_VARIABLE ||
               var_type == Shader::TGE_EFFECT_TYPE, "Expecting variable or type declaration");
    if(var_type == Shader::TGE_EFFECT_VARIABLE)
    {
        auto* var = var_node->extract<Shader::Variable>();
        auto* _type = var->getType();
        TGE_ASSERT(_type->getTypeEnum() == Shader::ElementType::Subroutine ||
                (_type->getTypeEnum() == Shader::ElementType::Array &&
                _type->getArrayElementType()->getTypeEnum() == Shader::ElementType::Subroutine),
                "Expecting subroutine type");
        m_RawImport.stream() << "subroutine uniform ";
        m_RawImport.visit(decl);
        
        UniformValueType uniform_value_type;
        size_t elem_size, array_size;
        ConvertSubroutine("", var->getType(), &uniform_value_type, &elem_size, &array_size);
        TGE_ASSERT(elem_size == sizeof(uint32), "Subroutine uniform should be 32-bit");
        m_Effect.addSubroutineUniform(Shader::BufferElement(m_Effect.getSubroutineUniformCount()*elem_size, uniform_value_type, var->getNodeName(), elem_size, array_size));
    }
    else if(var_type == Shader::TGE_EFFECT_TYPE)
    {
        m_RawImport.visit(decl);
    }
}

void Generator::visit(const Shader::Buffer* buffer)
{
    m_RawImport.visit(buffer);
    
    size_t offset = 0;
    Shader::BufferDescription buf_desc(buffer->getBufferType(), buffer->getNodeName());
    auto* list = buffer->getBody();
    for(auto iter = list->current(), iter_end = list->end(); iter != iter_end; ++iter)
    {
        auto* decl = iter->extract<Shader::Declaration>();
        auto* var = decl->getVariables()->extract<Shader::Variable>();
        
        size_t offset = 0;
        auto _type = var->getType(); 
        ConvertVariable("", var, &offset, &buf_desc);
        
    }
    m_Effect.addBuffer(buf_desc);
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

void Generator::visit(const Shader::Technique* _technique)
{
    // Techniques are used for gathering shaders in passes. The layout resembles the one in HLSL FX files.
    for(size_t i = 0, iend = m_Effect.getTechniqueCount(); i < iend; ++i)
        if(m_Effect.getTechnique(i).getName() == _technique->getNodeName())
        {
            Log(LogLevel::Error, "technique already specified ", _technique->getNodeName());
            m_Valid = false;
            return;
        }
    Shader::TechniqueDescription technique_desc(_technique->getNodeName());

    auto* technique_body = _technique->getBody();
    TGE_ASSERT(technique_body->current_front(), "Expected valid technique body");
    if(*technique_body->next())
    {
        Log(LogLevel::Error, "TODO: more than one pass is currently unsupported for this back-end");
        m_Valid = false;
        return;
    }
    auto* pass = technique_body->current_front()->extract<Shader::Pass>();
    Shader::PassDescription pass_desc(pass->getNodeName());
    for(auto j = pass->getBody()->current(); j != pass->getBody()->end(); ++j)
    {
        auto* function_call = j->extract<Shader::FunctionCall>();
        auto func_name = function_call->getNodeName();
        Shader::ShaderType current_shader_type;
        if(func_name == "SetVertexShader")
        {
            current_shader_type = Shader::ShaderType::VertexShader;
        }
        else if(func_name == "SetTessControlShader")
        {
            current_shader_type = Shader::ShaderType::TessellationControlShader;
        }
        else if(func_name == "SetTessEvaluationShader")
        {
            current_shader_type = Shader::ShaderType::TessellationEvaluationShader;
        }
        else if(func_name == "SetGeometryShader")
        {
            current_shader_type = Shader::ShaderType::GeometryShader;
        }
        else if(func_name == "SetFragmentShader")
        {
            current_shader_type = Shader::ShaderType::FragmentShader;
        }
        else if(func_name == "SetComputeShader")
        {
            current_shader_type = Shader::ShaderType::ComputeShader;
        }
        else
        {
            Log(LogLevel::Error, "Unexpected function call in technique \"", _technique->getNodeName(), "\": ", func_name);
            m_Valid = false;
            return;
        }

        auto args = function_call->getArguments();
        TGE_ASSERT(args && args->current_front()->getNodeType() == Shader::TGE_EFFECT_FUNCTION_CALL, "Expected to be function call");
        auto compile_phase = args->current_front()->extract<Shader::FunctionCall>();
        if(compile_phase->getFunction()->getNodeName() != "CompileShader")
        {
            Log(LogLevel::Error, "Unexpected function call in technique \"", _technique->getNodeName(), "\"; CompileShader expected: ", func_name);
            m_Valid = false;
            return;
        }
        auto cp_args = compile_phase->getArguments();
        TGE_ASSERT(cp_args->current_front(), "Expected valid arguments");
        auto profile_instance = cp_args->current_front()->extract<Shader::Variable>();
        auto shader_constructor_call = cp_args->next()->get()->current_front()->extract<Shader::ConstructorCall>();
        auto shader_type = shader_constructor_call->getType()->extract<Shader::ShaderDeclaration>();

        auto profile_name = profile_instance->getNodeName();
        string _version = TranslateGLSLVersion(profile_name);
        if(_version.empty())
        {
            m_Valid = false;
            return;
        }
        
        auto shader_name = shader_type->getNodeName();
        size_t i, iend;
        for(i = 0, iend = m_Effect.getShaderCount(); i < iend; ++i)
            if(m_Effect.getShader(i).getName() == shader_name)
                break;
        if(i == iend)
        {
            Log(LogLevel::Error, "Undefined shader: ", shader_name);
            m_Valid = false;
            return;
        }
        if(m_Effect.getShader(i).getShaderType() != current_shader_type)
        {
            Log(LogLevel::Error, "Shader type mismatches with the type that is being compiled: ", shader_name);
            m_Valid = false;
            return;
        }

        Shader::PassShaderDescription shader_desc(i, _version);
        pass_desc.addShader(shader_desc);
    }
    technique_desc.addPass(pass_desc);
    m_Effect.addTechnique(technique_desc);
}

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
        TGE_ASSERT(j->getNodeType() == Shader::TGE_EFFECT_BINARY_OPERATOR, "Expected binary operator statement as part of import statement");
        auto* binop = j->extract<Shader::BinaryOperator>();
        TGE_ASSERT(binop->getOperation() == Shader::TGE_EFFECT_ASSIGN, "Expected binary operator assignment operation as part of import statement");
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

void Generator::visit(const Shader::Type* type_stmt)
{
    TGE_ASSERT(type_stmt->getTypeEnum() == Shader::ElementType::Shader ||
               type_stmt->getTypeEnum() == Shader::ElementType::Struct ||
               type_stmt->getTypeEnum() == Shader::ElementType::Subroutine, "Unexpected top level type declaration");
    if(type_stmt->getTypeEnum() == Shader::ElementType::Subroutine)
    {
        m_RawImport.printSubroutine(type_stmt->extract<Shader::Subroutine>());
    }
    else
    {
        type_stmt->accept(this);
    }
}

void Generator::visit(const Shader::ShaderDeclaration* _shader)
{
    auto shader_type = _shader->getType();
    Shader::ShaderDescription shader_desc(_shader->getType(), _shader->getNodeName());
    std::stringstream ss;
    ss << m_RawImportStream.str();
    ShaderPrinter shader_printer(ss, m_BindingCounter, AST::TGE_AST_PRINT_LINE_LOCATION);
    shader_printer.setShaderType(shader_type);
    
    switch(_shader->getType())
    {
    case Shader::ShaderType::VertexShader:
    {
        shader_printer.stream() << "out gl_PerVertex {\n"
                                   "\tvec4 gl_Position;\n"
                                   "\tfloat gl_PointSize;\n"
                                   "\tfloat gl_ClipDistance[];\n"
                                   "};\n";
    } break;
    }
    
    for(auto j = _shader->getBody()->current(); j != _shader->getBody()->end(); ++j)
    {
        shader_printer.visit(j->getDeclarationLocation());
        
        TGE_ASSERT(*j, "Valid node expected. Bad parsing beforehand.");
        auto node_type = j->getNodeType();
        
        if(node_type != Shader::TGE_EFFECT_DECLARATION)
        {
            j->accept(&shader_printer);
        }
        else if(node_type == Shader::TGE_EFFECT_TYPE && j->extract<Shader::Type>()->getTypeEnum() == Shader::ElementType::Subroutine)
        {
            shader_printer.printSubroutine(j->extract<Shader::Type>()->extract<Shader::Subroutine>());
        }
        else
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

            // The point here is that we want to remove everything that is not valid
            // GLSL code.
            if(layout)
            {
                bool layout_started = false;
                for(auto k = layout->current(); k != layout->end(); ++k)
                {
                    auto* binop = k->extract<Shader::BinaryOperator>();
                    auto layout_id = binop->getLHSOperand()->extract<Shader::Identifier>()->getValue();
                    if(!layout_started)
                    {
                        layout_started = true;
                        shader_printer.stream() << "layout(";
                    }
                    else
                        shader_printer.stream() << ", ";
                    shader_printer.visit(binop);
                }
                if(layout_started)
                    shader_printer.stream() << ") ";
            }
            
            if(var->getStorage() == Shader::StorageQualifier::StructBuffer)
            {
                auto& os = shader_printer.stream();
                for(size_t i = 0, indentation = shader_printer.getIndentation(); i < indentation; ++i)
                    os << "\t";
                os << "layout(std430, binding = " << m_BindingCounter++ << ") buffer " << var->getNodeName() << "_StructBuffer "
                    << "{\n";
                for(size_t i = 0, indentation = shader_printer.getIndentation() + 1; i < indentation; ++i)
                    os << "\t";
                os << var->getType()->getNodeName() << " " << var->getNodeName() << "[]";
                for(size_t i = 0, indentation = shader_printer.getIndentation(); i < indentation; ++i)
                    os << "\t";
                os << "};\n";
            }
            else
            {
                switch(var->getInterpolation())
                {
                default: TGE_ASSERT(false, "Unknown interpolation type");
                case Shader::InterpolationQualifier::Default: break;
                case Shader::InterpolationQualifier::Smooth: shader_printer.stream() << "smooth "; break;
                case Shader::InterpolationQualifier::Flat: shader_printer.stream() << "flat "; break;
                case Shader::InterpolationQualifier::Noperspective: shader_printer.stream() << "noperspective "; break;
                }

                switch(var->getStorage())
                {
                case Shader::StorageQualifier::Default: break;
                case Shader::StorageQualifier::Const: shader_printer.stream() << "const "; break;
                case Shader::StorageQualifier::In: shader_printer.stream() << "in "; break;
                case Shader::StorageQualifier::CentroidIn: shader_printer.stream() << "centroid in "; break;
                case Shader::StorageQualifier::SampleIn: TGE_ASSERT(false, "sample out "); break;
                case Shader::StorageQualifier::Out: shader_printer.stream() << "out "; break;
                case Shader::StorageQualifier::CentroidOut: shader_printer.stream() << "centroid out "; break;
                case Shader::StorageQualifier::SampleOut: TGE_ASSERT(false, "sample in "); break;
                case Shader::StorageQualifier::InOut: TGE_ASSERT(false, "Unexpected storage format.");
                default:
                    TGE_ASSERT(false, "Unsupported storage type.");
                }

                var->getType()->accept(&shader_printer);
                shader_printer.stream() << " " << var->getNodeName();
            }
        }
        if(!j->isBlockStatement())
           shader_printer.stream() << ";\n";
     }
    
    string source;
    if(shader_printer.isDrawIDInUse() && shader_type != Shader::ShaderType::VertexShader)
    {
        std::stringstream interm;
        interm << "flat​ in int sys_DrawID_" << (uint32)shader_type << ";\n" << ss.str(); 
        source = ss.str();
    }
    else
    {
        source = ss.str();
    }
    
    shader_desc.appendContent(source);
    m_Effect.addShader(shader_desc);
}

static void PushSubroutineFunction(string name, Shader::EffectDescription* effect)
{
    for(size_t i = 0, iend = effect->getSubroutineFunctionCount(); i < iend; ++i)
    {
        if(effect->getSubroutineFunction(i) == name)
            return;
    }
    effect->addSubroutineFunction(name);   
}

void Generator::visit(const Shader::FunctionDefinition* func_def)
{
    m_RawImport.visit(func_def);
    auto func_decl = func_def->getDeclaration();
    if(func_decl->getSubroutineTypesBegin() != func_decl->getParametersEnd())
    {
        PushSubroutineFunction(func_def->getNodeName(), &m_Effect);
    }
}

void Generator::visit(const Shader::FunctionDeclaration* func_decl)
{
    m_RawImport.visit(func_decl);
    m_RawImportStream << ";\n";
    if(func_decl->getSubroutineTypesBegin() != func_decl->getParametersEnd())
    {
        PushSubroutineFunction(func_decl->getNodeName(), &m_Effect);
    }
}

bool LoadEffect(const string& filename, FileLoader* loader, Shader::EffectDescription& effect)
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

    GLFX::Generator _generator(loader);
    _generator.visit(root_node);
    effect = _generator.getEffect();
    return _generator.isValid();
}
}
}
