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
#include "tempest/graphics/opengl-backend/gl-config.hh"
#include "tempest/shader/gl-shader-generator.hh"
#include "tempest/utils/assert.hh"
#include "tempest/utils/logging.hh"

#define TGE_DEBUG_GLSL_APPEND_SOURCE

namespace Tempest
{
GLShaderType TranslateShaderType(Shader::ShaderType type)
{
    switch(type)
    {
    default: TGE_ASSERT(false, "Unknown shader type"); // fall-through
    case Shader::ShaderType::VertexShader: return GLShaderType::GL_VERTEX_SHADER;
    case Shader::ShaderType::TessellationControlShader: return GLShaderType::GL_TESS_CONTROL_SHADER;
    case Shader::ShaderType::TessellationEvaluationShader: return GLShaderType::GL_TESS_EVALUATION_SHADER;
    case Shader::ShaderType::GeometryShader: return GLShaderType::GL_GEOMETRY_SHADER;
    case Shader::ShaderType::FragmentShader: return GLShaderType::GL_FRAGMENT_SHADER;
    case Shader::ShaderType::ComputeShader: return GLShaderType::GL_COMPUTE_SHADER;
    }
}

GLShaderCompiler::GLShaderCompiler(uint32 settings)
    :   m_Settings(settings)
{
#ifndef TEMPEST_DISABLE_MDI
    if(!IsGLCapabilitySupported(TEMPEST_GL_CAPS_440))
#endif
    {
        m_Settings |= TEMPEST_DISABLE_MULTI_DRAW|TEMPEST_DISABLE_SSBO;
    }
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    if(!IsGLCapabilitySupported(TEMPEST_GL_CAPS_TEXTURE_BINDLESS))
#endif
    {
        m_Settings |= TEMPEST_DISABLE_TEXTURE_BINDLESS;
    }
}

GLShaderProgram* GLShaderCompiler::compileShaderProgram(const string& filename, FileLoader* file_loader,
                                                        const string* options, uint32 options_count)
{
    Shader::EffectDescription effect;
    auto status = GLFX::LoadEffect(filename, file_loader, options, options_count, m_Settings, effect);
    if(!status)
        return nullptr;
    
    auto prog = CreateScoped<GLuint>(glCreateProgram(), [](GLuint prog) { if(prog) glDeleteProgram(prog); });
    
    for(Shader::ShaderType i = Shader::ShaderType::VertexShader, iend = Shader::ShaderType::ShaderTypeCount;
        i != iend; ++reinterpret_cast<uint32&>(i))
    {
        auto* shader_desc = effect.getShader(i);
        if(shader_desc == nullptr)
            continue;
        auto source = shader_desc->getAdditionalOptions() + shader_desc->getContent();
        
        auto _type = static_cast<Shader::ShaderType>(i);
        GLuint shader_id = glCreateShader(TranslateShaderType(_type));
        
        GLint status;
        const char* cstr = source.c_str();
        glShaderSource(shader_id, 1, &cstr, nullptr);
        glCompileShader(shader_id);
        glGetShaderiv(shader_id, GLShaderParameter::GL_COMPILE_STATUS, &status);
        if(status == GL_FALSE)
        {
            GLint len;
            std::string error;

            glGetShaderiv(shader_id, GLShaderParameter::GL_INFO_LOG_LENGTH, &len);
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
    
    uint32 buf_count = effect.getBufferCount();
    uint32 res_table_count = buf_count;
    std::unique_ptr<ResourceTableDescription*[]> res_tables(new ResourceTableDescription*[res_table_count]);
    
    for(uint32 buffer_idx = 0; buffer_idx < buf_count; ++buffer_idx)
    {
        auto& buffer = effect.getBuffer(buffer_idx);
        auto& res_table = res_tables[buffer_idx];
        
        auto type = buffer.getBufferType();
        uint32 elem_count = buffer.getElementCount();
        
        res_table = CreatePackedData<ResourceTableDescription>(elem_count, buffer.getResiablePart(), buffer.getBufferName(), buffer_idx);
        res_table->BufferSize = 0;
        for(uint32 el_idx = 0; el_idx < elem_count; ++el_idx)
        {
            auto& elem_desc = buffer.getElement(el_idx);
            auto& uval = res_table->Uniforms.Values[el_idx];
            uval.Name = elem_desc.getElementName();
            uval.Type = elem_desc.getElementType();
            uval.ElementSize = static_cast<Tempest::uint16>(elem_desc.getElementSize());
            uval.ElementCount = static_cast<Tempest::uint16>(elem_desc.getElementCount());
            uval.Offset = static_cast<Tempest::uint32>(elem_desc.getBufferOffset());
            auto cur_end = static_cast<uint32>(uval.Offset + uval.ElementCount*uval.ElementSize);
            res_table->BufferSize = std::max(res_table->BufferSize, cur_end);
        }
        res_table->BufferSize = (res_table->BufferSize + 4 * sizeof(float) - 1) & ~(4 * sizeof(float) - 1);
        res_table->BufferSize -= static_cast<Tempest::uint32>(buffer.getResiablePart());
    }
    
    glLinkProgram(prog.get());
    
    GLint prog_status;
    glGetProgramiv(prog.get(), GLProgramParameter::GL_LINK_STATUS, &prog_status);
    TGE_ASSERT(prog_status != GL_FALSE, "Program compilation failed");
    if(prog_status == GL_FALSE)
    {
        GLint len;
        glGetProgramiv(prog.get(), GLProgramParameter::GL_INFO_LOG_LENGTH, &len);
        string error;
        error.resize(len);
        glGetProgramInfoLog(prog.get(), len, nullptr, &error.front());
        Log(LogLevel::Error, "Shader program link error: ", error);
        glDeleteProgram(prog.get());
            
#ifdef TGE_DEBUG_GLSL_APPEND_SOURCE
        for(Shader::ShaderType i = Shader::ShaderType::VertexShader, iend = Shader::ShaderType::ShaderTypeCount;
            i != iend; ++reinterpret_cast<uint32&>(i))
        {
            auto* shader_desc = effect.getShader(i);
            if(shader_desc == nullptr)
                continue;
            auto source = shader_desc->getAdditionalOptions() + shader_desc->getContent();

            auto _type = static_cast<Shader::ShaderType>(i);
            GLuint shader_id = glCreateShader(TranslateShaderType(_type));

            const char* header = "\n\nSource code\n"
                                 "=======================================\n\n";

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
            Log(LogLevel::Error, shader_type, header, source);
#endif
        }
        return nullptr;
    }
    
    return new GLShaderProgram(prog.release(), res_tables.release(), res_table_count);
}

void GLShaderCompiler::destroyRenderResource(GLShaderProgram* shader_program)
{
    delete shader_program;
}

FileDescription* GLShaderCompiler::compileBinaryBlob(const string& filename, FileLoader* file_loader,
                                                     const string* options, uint32 options_count)
{
    TGE_ASSERT(false, "Stub");
    return nullptr;
}

void GLShaderCompiler::destroyRenderResource(FileDescription* blob)
{
    TGE_ASSERT(false, "Stub");
}
}