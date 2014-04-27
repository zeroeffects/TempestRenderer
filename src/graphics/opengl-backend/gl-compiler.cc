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

GLShaderProgram* GLShaderCompiler::compileShaderProgram(const string& filename, FileLoader* file_loader)
{
    Shader::EffectDescription effect;
    auto status = GLFX::LoadEffect(filename, file_loader, effect);
    if(!status)
        return nullptr;
    
    if(effect.getTechniqueCount() == 0)
    {
        Log(LogLevel::Error, "Expecting at least one valid technique within file");
        return nullptr;
    }
    
    auto& technique = effect.getTechnique(0);
    
    if(technique.getPassCount() == 0)
    {
        Log(LogLevel::Error, "Expecting at least one valid pass within file");
        return nullptr;
    }
    
    auto& pass = technique.getPass(0);
    
    GLuint shader_prog = glCreateProgram();
    
    for(size_t i = 0, iend = pass.getAttachedShaderCount(); i < iend; ++i)
    {
        auto& shader_desc = pass.getAttachedShaderName(i);
        const Shader::ShaderDescription* shader = nullptr;
        for(size_t shader_idx = 0, end_shader_idx = effect.getShaderCount(); shader_idx < end_shader_idx; ++shader_idx)
        {
            shader = &effect.getShader(shader_idx);
            if(shader->getName() == shader_desc.getName())
                break;
        }
        auto source = shader_desc.getAdditionalOptions() + shader->getContent();
        
        auto _type = shader->getShaderType();
        GLuint id = glCreateShader(TranslateShaderType(_type));
        auto scoped = CreateAtScopeExit([&id](){ glDeleteShader(id); });
        
        GLint status;
        const char* cstr = source.c_str();
        glShaderSource(id, 1, &cstr, nullptr);
        glCompileShader(id);
        glGetShaderiv(id, GL_COMPILE_STATUS, &status);
        if(status == GL_FALSE)
        {
            GLint len;
            std::string error;

            glGetShaderiv(id, GL_INFO_LOG_LENGTH, &len);
            error.resize(len);
            glGetShaderInfoLog(id, len, nullptr, &error.front());
            
        #ifdef CE_DEBUG_GLSL_APPEND_SOURCE
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
            return nullptr;
        }
        
        glAttachShader(shader_prog, id);
    }
    
    GLint prog_status;
    glLinkProgram(shader_prog);
    glGetProgramiv(shader_prog, GL_LINK_STATUS, &prog_status);
    if(prog_status == GL_FALSE)
    {
        GLint len;
        glGetProgramiv(shader_prog, GL_INFO_LOG_LENGTH, &len);
        string error;
        error.resize(len);
        glGetProgramInfoLog(shader_prog, len, nullptr, &error.front());
        Log(LogLevel::Error, "Shader program link error: ", error);
    }
    return new GLShaderProgram(shader_prog);
}

void GLShaderCompiler::destroyRenderResource(GLShaderProgram* shader_program)
{
    delete shader_program;
}

FileDescription* GLShaderCompiler::compileBinaryBlob(const string& filename, FileLoader* file_loader)
{
    TGE_ASSERT(false, "Stub");
    return nullptr;
}

void GLShaderCompiler::destroyRenderResource(FileDescription* blob)
{
    TGE_ASSERT(false, "Stub");
}
}