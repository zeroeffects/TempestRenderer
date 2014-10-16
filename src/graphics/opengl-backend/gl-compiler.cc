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

#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-compiler.hh"
#include "tempest/graphics/opengl-backend/gl-shader.hh"
#include "tempest/shader/gl-shader-generator.hh"
#include "tempest/utils/assert.hh"
#include "tempest/utils/logging.hh"

#define TGE_DEBUG_GLSL_APPEND_SOURCE

namespace Tempest
{
GLenum TranslateShaderType(Shader::ShaderType type)
{
    switch(type)
    {
    default: TGE_ASSERT(false, "Unknown shader type"); // fall-through
    case Shader::ShaderType::VertexShader: return GL_VERTEX_SHADER;
    case Shader::ShaderType::TessellationControlShader: return GL_TESS_CONTROL_SHADER;
    case Shader::ShaderType::TessellationEvaluationShader: return GL_TESS_EVALUATION_SHADER;
    case Shader::ShaderType::GeometryShader: return GL_GEOMETRY_SHADER;
    case Shader::ShaderType::FragmentShader: return GL_FRAGMENT_SHADER;
    case Shader::ShaderType::ComputeShader: return GL_COMPUTE_SHADER;
    }
}

GLShaderProgram* GLShaderCompiler::compileShaderProgram(const string& filename, FileLoader* file_loader,
                                                        const string& technique_name, const string& pass_name)
{
    Shader::EffectDescription effect;
    auto status = GLFX::LoadEffect(filename, file_loader, effect);
    if(!status)
        return nullptr;
    
    const Shader::TechniqueDescription* technique;
    const Shader::PassDescription* pass;
    
    if(technique_name.empty()) 
    {
        if(effect.getTechniqueCount() == 0)
        {
            Log(LogLevel::Error, "Expecting at least one valid technique within file");
            return nullptr;
        }
        
        technique = &effect.getTechnique(0);
    }
    else
    {
        size_t i, iend;
        for(i = 0, iend = effect.getTechniqueCount(); i < iend; ++i)
        {
            technique = &effect.getTechnique(i);
            if(technique->getName() == technique_name)
                break;
        }
        if(i == iend)
        {
            Log(LogLevel::Error, "Unknown technique \"", technique_name, "\" within file");
            return nullptr;
        }
    }
    
    if(pass_name.empty())
    {
        if(technique->getPassCount() == 0)
        {
            Log(LogLevel::Error, "Expecting at least one valid pass within file");
            return nullptr;
        }
        
        pass = &technique->getPass(0);
    }
    else
    {
        size_t i, iend;
        for(i = 0, iend = technique->getPassCount(); i < iend; ++i)
        {
            pass = &technique->getPass(i);
            if(pass->getName() == pass_name)
                break;
        }
        if(i == iend)
        {
            Log(LogLevel::Error, "Unknown pass \"", pass_name, "\" as part of technique \"", technique_name, "\" within file");
            return nullptr;
        }
    }
    
    auto prog = CreateScoped<GLuint>(glCreateProgram(), [](GLuint prog) { if(prog) glDeleteProgram(prog); });
    
    for(size_t i = 0, iend = pass->getAttachedShaderCount(); i < iend; ++i)
    {
        auto& shader_desc = pass->getAttachedShaderName(i);
        const Shader::ShaderDescription* shader = nullptr;
        for(size_t shader_idx = 0, end_shader_idx = effect.getShaderCount(); shader_idx < end_shader_idx; ++shader_idx)
        {
            shader = &effect.getShader(shader_idx);
            if(shader->getName() == effect.getShader(shader_desc.getShaderIndex()).getName())
                break;
        }
        auto source = shader_desc.getAdditionalOptions() + shader->getContent();
        
        auto _type = shader->getShaderType();
        GLuint shader_id = glCreateShader(TranslateShaderType(_type));
        
        GLint status;
        const char* cstr = source.c_str();
        glShaderSource(shader_id, 1, &cstr, nullptr);
        glCompileShader(shader_id);
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);
        if(status == GL_FALSE)
        {
            GLint len;
            std::string error;

            glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &len);
            error.resize(len);
            glGetShaderInfoLog(shader_id, len, nullptr, &error.front());
            std::replace(error.begin(), error.end(), '\0', ' ');
            
        #ifdef TGE_DEBUG_GLSL_APPEND_SOURCE
            error += "\n\nSource code\n"
                    "=======================================\n\n";
            error += source;
        #endif
            string shader_type;
            switch(_type)
            {
            case Shader::ShaderType::VertexShader: shader_type = "Vertex shader:\n"; break;
            case Shader::ShaderType::TessellationControlShader: shader_type = "Tessellation control shader:\n"; break;
            case Shader::ShaderType::TessellationEvaluationShader: shader_type = "Tessellation evaluation shader:\n"; break;
            case Shader::ShaderType::GeometryShader: shader_type = "Geometry shader:\n"; break;
            case Shader::ShaderType::FragmentShader: shader_type = "Fragment shader:\n"; break;
            case Shader::ShaderType::ComputeShader: shader_type = "Compute shader:\n"; break;
            default: TGE_ASSERT(false, "Unsupported shader type."); break;
            }
            Log(LogLevel::Error, shader_type, error);
            glDeleteShader(shader_id);
            return nullptr;
        }
        
        glAttachShader(prog.get(), shader_id);
    }
    
    size_t buf_count = effect.getBufferCount();
    size_t res_table_count = buf_count;
    size_t subr_count = effect.getSubroutineUniformCount();
    size_t subr_func_count = effect.getSubroutineFunctionCount();
    if(subr_count)
        ++res_table_count;
    std::unique_ptr<ResourceTableDescription*[]> res_tables(new ResourceTableDescription*[res_table_count]);
    
    for(size_t buffer_idx = 0; buffer_idx < buf_count; ++buffer_idx)
    {
        auto& buffer = effect.getBuffer(buffer_idx);
        auto& res_table = res_tables[buffer_idx];
        
        auto type = buffer.getBufferType();
        auto elem_count = buffer.getElementCount();
        
        res_table = CreatePackedData<ResourceTableDescription>(elem_count, buffer.getResiablePart(), buffer.getBufferName(), buffer_idx);
        res_table->BufferSize = 0;
        for(size_t el_idx = 0; el_idx < elem_count; ++el_idx)
        {
            auto& elem_desc = buffer.getElement(el_idx);
            auto& uval = res_table->Uniforms.Values[el_idx];
            uval.Name = elem_desc.getElementName();
            uval.Type = elem_desc.getElementType();
            uval.ElementSize = elem_desc.getElementSize();
            uval.ElementCount = elem_desc.getElementCount();
            uval.Offset = elem_desc.getBufferOffset();
            auto cur_end = static_cast<uint32>(uval.Offset + uval.ElementCount*uval.ElementSize);
            res_table->BufferSize = std::max(res_table->BufferSize, cur_end);
        }
        res_table->BufferSize -= buffer.getResiablePart();
    }
    auto& subr_res_table = res_tables[buf_count];
    subr_res_table = CreatePackedData<ResourceTableDescription>(subr_func_count + subr_count, 0, "$Subroutines", 0);
    size_t offset = 0;
    for(size_t el_idx = 0; el_idx < subr_func_count; ++el_idx)
    {
        auto& subr_uniform = effect.getSubroutineFunction(el_idx);
        auto& uval = subr_res_table->Uniforms.Values[el_idx];
        uval.Name = subr_uniform;
        uval.Type = UniformValueType::SubroutineFunction;
        uval.ElementCount = 1;
        uval.Offset = el_idx;
    }
    subr_res_table->BufferSize = 0;
    for(size_t el_idx = 0, subr_idx = subr_count; el_idx < subr_count; ++el_idx, ++subr_idx)
    {
        auto& subr_func = effect.getSubroutineUniform(el_idx);
        auto& uval = subr_res_table->Uniforms.Values[subr_idx];
        uval.Name = subr_func.getElementName();
        uval.Type = subr_func.getElementType();
        uval.ElementSize = subr_func.getElementSize();
        uval.ElementCount = subr_func.getElementCount();
        uval.Offset = offset;
        auto cur_end = static_cast<uint32>(uval.Offset + uval.ElementCount*uval.ElementSize);
        subr_res_table->BufferSize = std::max(subr_res_table->BufferSize, cur_end);
    }
    
    glLinkProgram(prog.get());
    
    GLint prog_status;
    glGetProgramiv(prog.get(), GL_LINK_STATUS, &prog_status);
    if(prog_status == GL_FALSE)
    {
        GLint len;
        glGetProgramiv(prog.get(), GL_INFO_LOG_LENGTH, &len);
        string error;
        error.resize(len);
        glGetProgramInfoLog(prog.get(), len, nullptr, &error.front());
        Log(LogLevel::Error, "Shader program link error: ", error);
        glDeleteProgram(prog.get());
        return nullptr;
    }
    
    return new GLShaderProgram(prog.release(), res_tables.release(), res_table_count);
}

void GLShaderCompiler::destroyRenderResource(GLShaderProgram* shader_program)
{
    delete shader_program;
}

FileDescription* GLShaderCompiler::compileBinaryBlob(const string& filename, FileLoader* file_loader,
                                                     const string& technique_name, const string& pass_name)
{
    TGE_ASSERT(false, "Stub");
    return nullptr;
}

void GLShaderCompiler::destroyRenderResource(FileDescription* blob)
{
    TGE_ASSERT(false, "Stub");
}
}