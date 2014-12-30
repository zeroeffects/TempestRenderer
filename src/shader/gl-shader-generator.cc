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
#include "tempest/graphics/opengl-backend/gl-config.hh"
#include "tempest/shader/shader-convert-common.hh"
#include "tempest/shader/shader-ast.hh"
#include "tempest/shader/shader-driver.hh"
#include "tempest/parser/file-loader.hh"
#include "tempest/utils/patterns.hh"

#include <algorithm>
#include <unordered_map>

namespace Tempest
{
namespace GLFX
{
class ShaderPrinter;

bool PrintBuffer(AST::VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Shader::Buffer* buffer, size_t* ssbo_binding_counter, size_t* ubo_binding_counter);

class ShaderPrinter: public Shader::VisitorInterface
{
    // Because we have scope, we must initialize a new variables map
    AST::PrinterInfrastructure m_Printer;
    
    Shader::ShaderType         m_ShaderType = Shader::ShaderType::GenericShader;
    bool                       m_DrawIDInUse = false;
    
    bool                       m_Valid = true;
    uint32                     m_Settings;
    size_t                     m_SSBOBindingCounter;
    size_t                     m_UBOBindingCounter;
    
public:
    ShaderPrinter(std::ostream& os, size_t ssbo_binding_counter, size_t ubo_binding_counter, uint32 settings_flags, uint32 flags)
        :   m_Printer(os, flags),
            m_Settings(settings_flags),
            m_SSBOBindingCounter(ssbo_binding_counter),
            m_UBOBindingCounter(ubo_binding_counter) {}
    virtual ~ShaderPrinter()=default;

    std::ostream& stream() { return m_Printer.stream(); }
    size_t getIndentation() const { return m_Printer.getIndentation(); }

    bool isValid() const { return m_Valid; }

    void setShaderType(Shader::ShaderType shader_type) { m_ShaderType = shader_type; }
    bool isDrawIDInUse() const { return m_DrawIDInUse; }
    
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
    virtual void visit(const Shader::Import* _import) final { _import->printList(this, &m_Printer, "import"); }
    virtual void visit(const Shader::ShaderDeclaration* _shader) final { Shader::PrintNode(this, &m_Printer, _shader); }
    virtual void visit(const Shader::IfStatement* if_stmt) final { Shader::PrintNode(this, &m_Printer, if_stmt); }
    virtual void visit(const Shader::Type* type_stmt) final { Shader::PrintNode(this, type_stmt); }
    virtual void visit(const Shader::Buffer* buffer) final { m_Valid &= PrintBuffer(this, &m_Printer, buffer, &m_SSBOBindingCounter, &m_UBOBindingCounter); }
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
        TGE_ASSERT(var->getStorage() != Shader::StorageQualifier::StructBuffer, "Shouldn't happen");
    }
    
    Shader::PrintNode(this, &m_Printer, decl);
}

void ShaderPrinter::visit(const Shader::Variable* var)
{
    if(var->getNodeName() == "tge_DrawID")
    {
        if(m_Settings & TEMPEST_DISABLE_MULTI_DRAW)
        {
            m_Printer.stream() << "gl_InstanceID";
        }
        else
        {
            m_Printer.stream() << "gl_DrawIDARB";
        }
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

    string                     m_Version;

    size_t                     m_SSBOBindingCounter = TEMPEST_SSBO_START;
    size_t                     m_UBOBindingCounter = TEMPEST_UBO_START;
    bool                       m_Valid;
    Shader::EffectDescription& m_Effect;
    FileLoader*                m_FileLoader;
    uint32                     m_Settings;
public:
    Generator(Shader::EffectDescription& effect, FileLoader* include_loader, uint32);
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
    virtual void visit(const Shader::Import* _import) final;
    virtual void visit(const Shader::ShaderDeclaration* _shader) final;
    virtual void visit(const Shader::FunctionDefinition* func_def) final;
    virtual void visit(const Shader::FunctionDeclaration* func_decl) final;
    virtual void visit(const Shader::IfStatement* if_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Type* type_stmt) final;
    virtual void visit(const Shader::Buffer* buffer) final;
    // Some types that should not appear in AST
    virtual void visit(const Shader::IntermFuncNode*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FunctionSet*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::Expression*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::FuncDeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::DeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::VarDeclList*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }
    virtual void visit(const Shader::Value<Shader::ShaderType>*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code."); }

    bool isValid() const { return m_Valid & m_RawImport.isValid(); }
};

Generator::Generator(Shader::EffectDescription& effect, FileLoader* include_loader, uint32 settings)
    :   m_RawImport(m_RawImportStream, m_SSBOBindingCounter, m_UBOBindingCounter, m_Settings, AST::TGE_AST_PRINT_LINE_LOCATION),
        m_Effect(effect),
        m_Valid(true),
        m_FileLoader(include_loader),
        m_Settings(settings)
{
    uint32 min_version = 420;
    if((m_Settings & TEMPEST_DISABLE_SSBO) == 0)
    {
        min_version = std::max(min_version, 430U);
    }
    if((m_Settings & TEMPEST_DISABLE_MULTI_DRAW) == 0)
    {
        m_Version += "#extension GL_ARB_shader_draw_parameters : require\n";
    }
    if((m_Settings & TEMPEST_DISABLE_TEXTURE_BINDLESS) == 0)
    {
        m_Version += "#extension GL_ARB_bindless_texture : require\n";
    } // TODO: Bindless and disabled SSBO
    std::stringstream ss;
    ss << "#version " << min_version << "\n";

    m_Version = ss.str() + m_Version + "\n";
}

Generator::~Generator() {}

bool PrintBuffer(AST::VisitorInterface* visitor, AST::PrinterInfrastructure* printer, const Shader::Buffer* buffer, size_t* ssbo_binding_counter, size_t* ubo_binding_counter)
{
    bool status = true;
    switch(buffer->getBufferType())
    {
    default: TGE_ASSERT(false, "Unknown buffer type");
    case Shader::BufferType::Constant:
    {
        for(size_t i = 0, indentation = printer->getIndentation(); i < indentation; ++i)
            printer->stream() << "\t";
        printer->stream() << "layout(std140, binding = " << (*ubo_binding_counter)++ << ") uniform " << buffer->getNodeName()
                          << "{\n";
        auto* list = buffer->getBody();
        visitor->visit(list);
        printer->stream() << "};\n";
    } break;
    case Shader::BufferType::Regular:
    {
        for(size_t i = 0, indentation = printer->getIndentation(); i < indentation; ++i)
            printer->stream() << "\t";
        printer->stream() << "layout(std430, binding = " << (*ssbo_binding_counter)++ << ") buffer " << buffer->getNodeName()
                          << "{\n";
        auto* list = buffer->getBody();
        visitor->visit(list);
        printer->stream() << "};\n";
    } break;
    case Shader::BufferType::Resource:
    {
        for(size_t i = 0, indentation = printer->getIndentation(); i < indentation; ++i)
            printer->stream() << "\t";
        printer->stream() << "layout(std140, binding = " TO_STRING(TEMPEST_RESOURCE_BUFFER) ") uniform " << buffer->getNodeName()
            << "{\n";
        auto* list = buffer->getBody();
        for(auto iter = list->current(), iter_end = list->end(); iter != iter_end; ++iter)
        {
            TGE_ASSERT(iter->getNodeType() == Shader::TGE_EFFECT_DECLARATION, "Expected declaration");
            auto* vars = iter->extract<Shader::Declaration>()->getVariables();
            auto check_variable = [](const Shader::Variable* var)
            {
                if(var->getType()->getTypeEnum() != Shader::ElementType::Sampler)
                {
                    Log(LogLevel::Error, "Top-level samplers are only allowed in resource buffer for compatibility reasons.");
                    return false;
                }
                return true;
            };

            if(vars->getNodeType() == Shader::TGE_EFFECT_VARIABLE)
            {
                auto var = vars->extract<Shader::Variable>();
                status &= check_variable(var);
            }
            else if(vars->getNodeType() == AST::TGE_AST_LIST_ELEMENT)
            {
                auto sub_list = vars->extract<AST::List>();
                TGE_ASSERT(sub_list->current()->getNodeType() == Shader::TGE_EFFECT_VARIABLE, "Expecting variables within declaration list");
                auto sub_start = sub_list->current();
                auto* var = sub_start->extract<Shader::Variable>();
                status &= check_variable(var);
            }
        }
        visitor->visit(list);
        printer->stream() << "};\n";
    } break;
    }
    return status;
}

static void ProcessStructBuffer(ShaderPrinter* visitor, uint64 settings, const Shader::Variable* var,
                                size_t* ssbo_binding_counter, size_t* ubo_binding_counter, Shader::EffectDescription* effect)
{
    auto size = ConvertStructBuffer(var, effect);

    size_t elems = (1 << 16) / size;

    auto& os = visitor->stream();
    for(size_t i = 0, indentation = visitor->getIndentation(); i < indentation; ++i)
        os << "\t";
    if(settings & TEMPEST_DISABLE_SSBO)
        os << "layout(std140, binding = " TO_STRING(TEMPEST_GLOBALS_BUFFER) ") uniform ";
    else
        os << "layout(std430, binding = " TO_STRING(TEMPEST_GLOBALS_BUFFER) ") buffer ";
    os << var->getNodeName() << "_StructBuffer "
        << "{\n";
    for(size_t i = 0, indentation = visitor->getIndentation() + 1; i < indentation; ++i)
        os << "\t";
    os << var->getType()->getArrayElementType()->getNodeName() << " " << var->getNodeName();
    if(settings & TEMPEST_DISABLE_SSBO)
        os << "[" << elems << "];\n";
    else
        os << "[];\n";
    for(size_t i = 0, indentation = visitor->getIndentation(); i < indentation; ++i)
        os << "\t";
    os << "}";
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
            ProcessStructBuffer(&m_RawImport, m_Settings, var, &m_SSBOBindingCounter, &m_UBOBindingCounter, &m_Effect);
        }
        else
        {
            Log(LogLevel::Error, "Unexpected variable declaration ", var->getNodeName(), ".");
            m_Valid = false;
            return;
        }
    }
}

void Generator::visit(const Shader::Buffer* buffer)
{
    m_RawImport.visit(buffer);
    
    ConvertBuffer(buffer, &m_Effect);
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
    TGE_ASSERT(type_stmt->getTypeEnum() == Shader::ElementType::Struct, "Unexpected top level type declaration");
    type_stmt->accept(this);
}

void Generator::visit(const Shader::ShaderDeclaration* _shader)
{
    auto shader_type = _shader->getType();
    std::unique_ptr<Shader::ShaderDescription> shader_desc(new Shader::ShaderDescription);
    std::stringstream ss;
    ss << m_RawImportStream.str();
    ShaderPrinter shader_printer(ss, m_SSBOBindingCounter, m_UBOBindingCounter, m_Settings, AST::TGE_AST_PRINT_LINE_LOCATION);
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
                ProcessStructBuffer(&shader_printer, m_Settings, var, &m_SSBOBindingCounter, &m_UBOBindingCounter, &m_Effect);
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

    shader_desc->setAdditionalOptions(m_Version);
    shader_desc->appendContent(source);
    if(!m_Effect.trySetShader(shader_type, shader_desc.release()))
    {
        Log(LogLevel::Error, "Duplicate shader types: ", ConvertShaderTypeToText(_shader->getType()), "\n"
            "Try using \"options\" to eliminate some of the shaders.");
        m_Valid = false;
        return;
    }

    m_Valid &= shader_printer.isValid();
}

void Generator::visit(const Shader::FunctionDefinition* func_def)
{
    m_RawImport.visit(func_def);
}

void Generator::visit(const Shader::FunctionDeclaration* func_decl)
{
    m_RawImport.visit(func_decl);
}

bool LoadEffect(const string& filename, FileLoader* loader, uint32 flags, Shader::EffectDescription& effect)
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

    GLFX::Generator _generator(effect, loader, flags);
    _generator.visit(root_node);
    return _generator.isValid();
}
}
}
