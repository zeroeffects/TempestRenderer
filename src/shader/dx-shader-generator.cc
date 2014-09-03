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
#include "tempest/shader/file-loader.hh"
#include "tempest/shader/shader-ast.hh"
#include "tempest/shader/shader-driver.hh"
#include "tempest/parser/ast.hh"

#include <unordered_map>

namespace Tempest
{
namespace DXFX
{
typedef std::unordered_map<string, string> SamplerTextureAssociation;
typedef std::vector<string> ShaderSignature;

class ShaderPrinter: public Shader::VisitorInterface
{
    AST::PrinterInfrastructure m_Printer;
    
    bool                       m_Valid;
    
    SamplerTextureAssociation  m_SamplerAssoc;
    ShaderSignature            m_InputSignature;
    ShaderSignature            m_OutputSignature;
    
    typedef bool (ShaderPrinter::*TranslationFunction)(const Shader::FunctionCall* func_call);
    std::unordered_map<string, TranslationFunction> m_FunctionTranslator;
public:
    ShaderPrinter(std::ostream& os, size_t flags);
    virtual ~ShaderPrinter();

    void setSamplerTextureAssociation(string texture, string sampler)
    {
        m_SamplerAssoc[texture] = sampler;
    }
    
    void setInputSignature(ShaderSignature signature) { m_InputSignature = signature; }
    void setOutputSignature(ShaderSignature signature) { m_OutputSignature = signature; }
    
    void setIndentation(size_t indentation) { m_Printer.setIndentation(indentation); }
    
    std::ostream& stream() { return m_Printer.stream(); }

    bool isValid() const { return m_Valid; }
    
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
    virtual void visit(const Shader::MatrixType* matrix_type) final { Shader::PrintNode(&m_Printer, matrix_type); }
    virtual void visit(const Shader::SamplerType* sampler_type) final { Shader::PrintNode(&m_Printer, sampler_type); }
    virtual void visit(const Shader::ArrayType* array_type) final { Shader::PrintNode(this, &m_Printer, array_type); }
    virtual void visit(const Shader::Variable* var) final;
    virtual void visit(const Shader::Declaration* decl) final { Shader::PrintNode(this, &m_Printer, decl); }
    virtual void visit(const Shader::MemberVariable* mem_var) final { Shader::PrintNode(this, &m_Printer, mem_var); }
    virtual void visit(const Shader::ArrayElementVariable* array_elem) final { Shader::PrintNode(this, &m_Printer, array_elem); }
    virtual void visit(const Shader::InvariantDeclaration* invar_decl) final;
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
    virtual void visit(const Shader::Sampler* _sampler) final { _sampler->printList(this, &m_Printer, "sampler"); }
    virtual void visit(const Shader::Import* _import) final { _import->printList(this, &m_Printer, "import"); }
    virtual void visit(const Shader::Shader* _shader) final { Shader::PrintNode(this, &m_Printer, _shader); }
    virtual void visit(const Shader::CompiledShader* compiled_shader) final { Shader::PrintNode(&m_Printer, compiled_shader); }
    virtual void visit(const Shader::Pass* _pass) final { _pass->printList(this, &m_Printer, "pass"); }
    virtual void visit(const Shader::IfStatement* if_stmt) final { Shader::PrintNode(this, &m_Printer, if_stmt); }
    virtual void visit(const Shader::Type* type_stmt) final { Shader::PrintNode(this, type_stmt); }
    virtual void visit(const Shader::Buffer* buffer) final { TGE_ASSERT(false, "Currently unsupported"); }
    // Some types that should not appear in AST
    virtual void visit(const Shader::FunctionSet*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Expression*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::FuncDeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::DeclarationInfo*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::VarDeclList*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    virtual void visit(const Shader::Value<Shader::ShaderType>*) final { TGE_ASSERT(false, "Unsupported. Probably you have made a mistake. Check your code"); }
    
private:
    bool TranslateTexelFetch(const Shader::FunctionCall* func_call);
};

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

ShaderPrinter::ShaderPrinter(std::ostream& os, size_t flags)
    :   m_Printer(os, flags),
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

void ShaderPrinter::visit(const Shader::Variable* var)
{
    auto var_name = var->getNodeName();
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
                m_Printer.stream() << "shader_out__.Position";
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
    std::stringstream          m_RawImportStream;
    Shader::Printer            m_RawImport;

    bool                       m_Valid;
    Shader::EffectDescription  m_Effect;
    FileLoader*                m_FileLoader;
public:
    Generator(FileLoader* include_loader);
    virtual ~Generator();

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
    virtual void visit(const Shader::Declaration* decl) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
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
    virtual void visit(const Shader::Sampler* _sampler) final;
    virtual void visit(const Shader::Import* _import) final;
    virtual void visit(const Shader::Shader* _shader) final;
    virtual void visit(const Shader::FunctionDefinition* func_def) final;
    virtual void visit(const Shader::FunctionDeclaration* func_decl) final;
    virtual void visit(const Shader::CompiledShader* compiled_shader) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Pass* _pass) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::IfStatement* if_stmt) final { TGE_ASSERT(false, "Unexpected. This node shouldn't appear at top level"); }
    virtual void visit(const Shader::Type* type_stmt) final;
    virtual void visit(const Shader::Buffer* buffer) final;
    // Some types that should not appear in AST
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
    :   m_RawImport(m_RawImportStream, AST::TGE_AST_PRINT_LINE_LOCATION),
        m_Valid(true),
        m_FileLoader(include_loader) {}

Generator::~Generator() {}

// TODO: I sort of hate these classifications because they don't show what features I really need. And I don't need them in the first place;
//       it is quite easy to figure it out just by matching functions and features. Yeah, well. It is good for enforcing particular shader model,
//       but that's more of a global option then per shader thing.
string ConvertGLSLVersionToHLSL(Shader::ShaderType _type, const string& profile_name)
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
    
    if(profile_name == "glsl_1_4_0")
        return shader_type + "3_0";
    else if(profile_name == "glsl_1_5_0")
        return shader_type + "4_0"; // Because geometry shaders.
    else if(profile_name == "glsl_3_3_0")
        return shader_type + "4_0";
    else if(profile_name == "glsl_4_0_0")
        return shader_type + "5_0";
    else if(profile_name == "glsl_4_1_0")
        return shader_type + "5_0";
    else if(profile_name == "glsl_4_2_0")
        return shader_type + "5_0";
    
    Log(LogLevel::Error, "unknown profile: ", profile_name);
    return "";
}

void Generator::visit(const Shader::Technique* _technique)
{
    // Techniques are used for gathering shaders in passes. The layout resembles the one in HLSL FX files.
    for(size_t i = 0, iend = m_Effect.getTechniqueCount(); i < iend; ++i)
        if(m_Effect.getTechnique(i).getName() == _technique->getNodeName())
        {
            Log(LogLevel::Error, "technique already specified: ", _technique->getNodeName());
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
            current_shader_type = Shader::ShaderType::VertexShader;
        else if(func_name == "SetTessControlShader")
            current_shader_type = Shader::ShaderType::TessellationControlShader;
        else if(func_name == "SetTessEvaluationShader")
            current_shader_type = Shader::ShaderType::TessellationEvaluationShader;
        else if(func_name == "SetGeometryShader")
            current_shader_type = Shader::ShaderType::GeometryShader;
        else if(func_name == "SetFragmentShader")
            current_shader_type = Shader::ShaderType::FragmentShader;
        else if(func_name == "SetComputeShader")
            current_shader_type = Shader::ShaderType::ComputeShader;
        else
        {
            Log(LogLevel::Error, "Unexpected function call in technique \"", _technique->getNodeName(), "\": ", func_name);
            m_Valid = false;
            return;
        }


        auto args = function_call->getArguments();
        TGE_ASSERT(args && args->current_front()->getNodeType() == Shader::TGE_EFFECT_FUNCTION_CALL, "The declaration is expected to be function call");
        auto compile_phase = args->current_front()->extract<Shader::FunctionCall>();
        if(compile_phase->getFunction()->getNodeName() != "CompileShader")
        {
            Log(LogLevel::Error, "Unexpected function call in technique \"",  _technique->getNodeName(), "\"; CompileShader expected: ", func_name);
            m_Valid = false;
            return;
        }
        auto cp_args = compile_phase->getArguments();
        TGE_ASSERT(cp_args->current_front(), "Expected valid arguments");
        auto profile_instance = cp_args->current_front()->extract<Shader::Variable>();
        auto shader_constructor_call = cp_args->next()->get()->current_front()->extract<Shader::ConstructorCall>();
        auto shader_type = shader_constructor_call->getType()->extract<Shader::Shader>();

        auto profile_name = profile_instance->getNodeName();
        string _version = ConvertGLSLVersionToHLSL(current_shader_type, profile_name);
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

void Generator::visit(const Shader::Buffer* buffer)
{
    m_RawImportStream << "cbuffer " << buffer->getNodeName() << " {";
        m_RawImport.visit(buffer->getBody());
    m_RawImportStream << "};";
}

void Generator::visit(const AST::ListElement* lst)
{
    for(auto i = lst->current(), iend = lst->end(); i != iend; ++i)
        i->accept(this);
}

void Generator::visit(const Shader::Type* type_stmt)
{
    TGE_ASSERT(type_stmt->getTypeEnum() == Shader::ElementType::Shader, "Unexpected top level type declaration");
    type_stmt->accept(this);
}

// Same as GLFX
void Generator::visit(const Shader::Sampler* _sampler)
{
    // Sampler definitions in HLSL FX style, but with OpenGL parameter values to make it easier for people that
    // are used to them.
    auto sampler_body = _sampler->getBody();
    TGE_ASSERT(sampler_body, "Sampler should have a valid body"); // TODO: Hm, not exactly valid assumption
    Shader::SamplerDescription fxsampler(_sampler->getNodeName());
    for(auto j = sampler_body->current(); j != sampler_body->end(); ++j)
    {
        if(j->getNodeType() != Shader::TGE_EFFECT_BINARY_OPERATOR)
        {
            Log(LogLevel::Error, "Unexpected node type in sampler: ", _sampler->getNodeName());
            return;
        }
        auto* binop = j->extract<Shader::BinaryOperator>();
        string state_name = binop->getLHSOperand()->extract<Shader::Identifier>()->getValue(),
               state_value = binop->getRHSOperand()->extract<Shader::Identifier>()->getValue();
        Shader::ParameterDescription param(state_name, state_value);
        fxsampler.addParameter(param);
    }
    m_Effect.addSampler(fxsampler);
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

// BUG: I am quite sure that I have misinterpreted samplerExt in some of my other code.

void Generator::visit(const Shader::Shader* _shader)
{
    Shader::ShaderDescription shader_desc(_shader->getType(), _shader->getNodeName());

    ShaderSignature             input_signature,
                                output_signature,
                                sampler_state;

    std::stringstream           in_signature_ss,
                                out_signature_ss,
                                ss;
                                
    // The actual trick out here is that we want the common stuff to be printed through
    // the extended version which accumulates all associations.
    ShaderPrinter   common_printer(ss, AST::TGE_AST_PRINT_LINE_LOCATION),
                    in_printer(in_signature_ss, AST::TGE_AST_PRINT_LINE_LOCATION),
                    out_printer(out_signature_ss, AST::TGE_AST_PRINT_LINE_LOCATION);
    const Shader::FunctionDefinition* entrypoint = nullptr;
    
    if(_shader->getType() == Shader::ShaderType::VertexShader)
    {
        output_signature.push_back("gl_Position");
        out_signature_ss << "shader_input_signature__ shader_in__\n"
                            "{\n"
                            "\tfloat4 Position: SV_POSITION;\n";
    }
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
            
            string prefix, suffix;
            if(layout)
            {
                for(auto k = layout->current(); k != layout->end(); ++k)
                {
                    auto* binop = k->extract<Shader::BinaryOperator>();
                    auto layout_id = binop->getLHSOperand()->extract<AST::Identifier>()->getValue();
                    if(layout_id == "semanticExt")
                    {
                        auto _storage = var->getStorage();
                        if(_storage == Shader::TGE_EFFECT_DEFAULT_STORAGE || _storage == Shader::TGE_EFFECT_CONST_STORAGE)
                        {
                            Log(LogLevel::Error, var_node->getDeclarationLocation(), ": Semantic can't be applied on variables that are not associated with input or output signature: ", var->getNodeName());
                            m_Valid = false;
                            return;
                        }
                            
                        if(!suffix.empty())
                        {
                            Log(LogLevel::Error, "There should not be more than one semantic values per object");
                            m_Valid = false;
                            return;
                        }
                        
                        auto* semantic_name_val = binop->getRHSOperand()->extract<AST::Identifier>();
                        if(_storage == Shader::TGE_EFFECT_IN_STORAGE || _storage == Shader::TGE_EFFECT_CENTROID_IN_STORAGE)
                        {
                            Shader::InputParameter param(var->getType()->getTypeEnum(), var->getNodeName(), semantic_name_val->getValue());
                            shader_desc.addInputParameter(param);
                        }
                        string semantic = semantic_name_val->getValue();
                        suffix = ": " + semantic;
                    }
                    else
                    {
                        Log(LogLevel::Error, k->getDeclarationLocation(), ": Unsupported layout qualifier: ", layout_id);
                        return;
                    }
                }
            }

            std::stringstream* current_ss = &ss;
            ShaderPrinter* current_printer = &common_printer;
            switch(var->getStorage())
            {
            case Shader::TGE_EFFECT_DEFAULT_STORAGE: j->printLocation(ss); break;
            case Shader::TGE_EFFECT_CONST_STORAGE: j->printLocation(ss); prefix = "const "; break;
            case Shader::TGE_EFFECT_IN_STORAGE:
            {
                if(input_signature.empty())
                    in_signature_ss << "struct shader_input_signature__\n{" << std::endl;
                current_ss = &in_signature_ss;
                j->printLocation(in_signature_ss);
                in_signature_ss << "\t";
                current_printer = &in_printer;
                input_signature.push_back(var->getNodeName());
            } break;
            case Shader::TGE_EFFECT_CENTROID_IN_STORAGE: 
            {
                if(input_signature.empty())
                    in_signature_ss << "struct shader_input_signature__\n{" << std::endl;
                prefix = "centroid ";
                current_ss = &in_signature_ss;
                j->printLocation(in_signature_ss);
                in_signature_ss << "\t";
                current_printer = &in_printer;
                input_signature.push_back(var->getNodeName());
            } break;
            case Shader::TGE_EFFECT_SAMPLE_IN_STORAGE: TGE_ASSERT(false, "Unsupported"); break;
            case Shader::TGE_EFFECT_OUT_STORAGE:
            {
                if(output_signature.empty())
                    out_signature_ss << "struct shader_output_signature__\n{" << std::endl;
                current_ss = &out_signature_ss;
                j->printLocation(out_signature_ss);
                out_signature_ss << "\t";
                current_printer = &out_printer;
                output_signature.push_back(var->getNodeName());
            } break;
            case Shader::TGE_EFFECT_CENTROID_OUT_STORAGE:
            {
                if(output_signature.empty())
                    out_signature_ss << "struct shader_output_signature__\n{" << std::endl;
                prefix = "centroid ";
                current_ss = &out_signature_ss;
                j->printLocation(out_signature_ss);
                out_signature_ss<< "\t";
                current_printer = &out_printer;
                output_signature.push_back(var->getNodeName());
            } break;
            case Shader::TGE_EFFECT_SAMPLE_OUT_STORAGE: TGE_ASSERT(false, "Unsupported"); break;
            case Shader::TGE_EFFECT_INOUT_STORAGE: TGE_ASSERT(false, "Unexpected storage format."); break;
            default:
                TGE_ASSERT(false, "Unsupported storage type.");
            }
            
            if(!prefix.empty())
                *current_ss << prefix << " ";
            var->getType()->accept(current_printer);
            *current_ss << " " << var->getNodeName() << suffix << ";\n";
        }
        else
        {
            j->printLocation(ss);
            j->accept(&common_printer);
            if(!j->isBlockStatement())
                ss << ";\n";
        }
    }

    if(!input_signature.empty())
        in_signature_ss << "};" << std::endl;
    if(!output_signature.empty())
        out_signature_ss << "};" << std::endl;

    if(entrypoint == nullptr)
    {
        Log(LogLevel::Error, "Shader must contain valid entrypoint: ", _shader->getNodeName());
        m_Valid = false;
        return;
    }
    
    for(auto i = sampler_state.begin(), iend = sampler_state.end(); i != iend; ++i)
        ss << "SamplerState " << *i << ";\n";
    ss << in_signature_ss.str() << out_signature_ss.str();
    
    if(output_signature.empty())
        ss << "void ";
    else
        ss << "shader_output_signature__ ";
    
    ss << _shader->getNodeName() << "(";
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
    
    shader_desc.appendContent(ss.str());
    m_Effect.addShader(shader_desc);
    
    if(!common_printer.isValid() ||
       !in_printer.isValid() ||
       !out_printer.isValid())
    {
        m_Valid = false;
        return;
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

    DXFX::Generator _generator(loader);
    _generator.visit(root_node);
    effect = _generator.getEffect();
    return _generator.isValid();
}
}
}
