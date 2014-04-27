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
#include "tempest/shader/file-loader.hh"
#include "tempest/utils/patterns.hh"

namespace Tempest
{
namespace GLFX
{
class Generator: public Shader::VisitorInterface
{
    std::stringstream          m_RawImportStream;
    Shader::Printer            m_RawImport;

    bool                       m_Valid;
    Shader::EffectDescription  m_Effect;
    FileLoader*             m_FileLoader;
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

string TranslateGLSLVersion(const string& profile_name)
{
    // I prefer this over an lazy hack that copies numbers because it is much safer.
    if(profile_name == "glsl_1_4_0")
        return "#version 140\n";
    else if(profile_name == "glsl_1_5_0")
        return "#version 150\n";
    else if(profile_name == "glsl_3_3_0")
        return "#version 330\n";
    else if(profile_name == "glsl_4_0_0")
        return "#version 400\n";
    else if(profile_name == "glsl_4_1_0")
        return "#version 410\n";
    else if(profile_name == "glsl_4_2_0")
        return "#version 420\n";
    
    Log(LogLevel::Error, "Unknown profile: ", profile_name);
    return "";
}

void Generator::visit(const Shader::Buffer* buffer)
{
    m_RawImportStream << "buffer " << buffer->getNodeName() << " {\n";
        m_RawImport.visit(buffer->getBody());
    m_RawImportStream << "};";
}

void Generator::visit(const AST::ListElement* lst)
{
    for(auto i = lst->current(), iend = lst->end(); i != iend; ++i)
        i->accept(this);
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
        auto shader_type = shader_constructor_call->getType()->extract<Shader::Shader>();

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

        Shader::PassShaderDescription shader_desc(shader_name, _version);
        pass_desc.addShader(shader_desc);
    }
    technique_desc.addPass(pass_desc);
    m_Effect.addTechnique(technique_desc);
}

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
            m_Valid = false;
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
    TGE_ASSERT(type_stmt->getTypeEnum() == Shader::TGE_EFFECT_SHADER_TYPE, "Unexpected top level type declaration");
    type_stmt->accept(this);
}

void Generator::visit(const Shader::Shader* _shader)
{
    Shader::ShaderDescription shader_desc(_shader->getType(), _shader->getNodeName());
    std::stringstream ss;
    Shader::Printer shader_printer(ss, AST::TGE_AST_PRINT_LINE_LOCATION);
    for(auto j = _shader->getBody()->current(); j != _shader->getBody()->end(); ++j)
    {
        j->printLocation(ss);
        
        TGE_ASSERT(*j, "Valid node expected. Bad parsing beforehand.");
        if(j->getNodeType() != Shader::TGE_EFFECT_DECLARATION)
            j->accept(&shader_printer);
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
                        
                        auto* semantic_name_val = binop->getRHSOperand()->extract<AST::Identifier>();
                        Shader::InputParameter param(var->getType()->getTypeEnum(), var->getNodeName(), semantic_name_val->getValue());
                        shader_desc.addInputParameter(param);
                    }
                    else if(layout_id == "samplerExt")
                    {
                        if(var->getType()->getTypeEnum() != Shader::TGE_EFFECT_SAMPLER_TYPE)
                        {
                            Log(LogLevel::Error, var_node->getDeclarationLocation(), ": Sampler layout qualifier applied on variable that is not a texture: ", var->getNodeName(), " of type: ", var->getType());
                            m_Valid = false;
                            return;
                        }
                        
                        auto* sampler = binop->getRHSOperand()->extract<Shader::Sampler>();
                        shader_desc.addSampler(sampler->getNodeName(), var->getNodeName());
                    }
                    else
                    {
                        if(!layout_started)
                        {
                            layout_started = true;
                            shader_printer.stream() << "layout(";
                        }
                        else
                            shader_printer.stream() << ", ";
                        shader_printer.visit(binop);
                    }
                }
                if(layout_started)
                    ss << ") ";
            }
            switch(var->getStorage())
            {
            case Shader::TGE_EFFECT_DEFAULT_STORAGE: break;
            case Shader::TGE_EFFECT_CONST_STORAGE: ss << "const "; break;
            case Shader::TGE_EFFECT_IN_STORAGE: ss << "in "; break;
            case Shader::TGE_EFFECT_CENTROID_IN_STORAGE:ss << "centroid in ";break;
            case Shader::TGE_EFFECT_SAMPLE_IN_STORAGE: TGE_ASSERT(false, "sample out "); break;
            case Shader::TGE_EFFECT_OUT_STORAGE: ss << "out "; break;
            case Shader::TGE_EFFECT_CENTROID_OUT_STORAGE: ss << "centroid out "; break;
            case Shader::TGE_EFFECT_SAMPLE_OUT_STORAGE: TGE_ASSERT(false, "sample in "); break;
            case Shader::TGE_EFFECT_INOUT_STORAGE: TGE_ASSERT(false, "Unexpected storage format.");
            default:
                TGE_ASSERT(false, "Unsupported storage type.");
            }
            
            var->getType()->accept(&shader_printer);
            ss << " " << var->getNodeName();
        }
        if(!j->isBlockStatement())
           ss << ";\n";
     }
    
    shader_desc.appendContent(ss.str());
    m_Effect.addShader(shader_desc);
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

    GLFX::Generator _generator(loader);
    _generator.visit(root_node);
    effect = _generator.getEffect();
    return _generator.isValid();
}
}
}
