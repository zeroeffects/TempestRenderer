/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2010-2014 Zdravko Velinov
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

#ifndef _GL_SHADER_COMPILER_HH_
#define _GL_SHADER_COMPILER_HH_

#include <cstdint>
#include "tempest/graphics/rendering-definitions.hh"

namespace Tempest
{
class FileLoader;
struct FileDescription;
class GLShaderProgram;
class GLRenderingBackend;

class GLShaderCompiler
{
    uint32_t m_Settings;
public:
    typedef GLShaderProgram ShaderProgramType;
    
    GLShaderCompiler(uint32_t settings = 0);
    ~GLShaderCompiler() = default;

    void setSettings(uint32_t settings) { m_Settings = settings; }

    GLShaderProgram* compileShaderProgram(const std::string& filename, FileLoader* file_loader,
                                          const std::string* options = nullptr, uint32_t options_count = 0);
    void destroyRenderResource(GLShaderProgram* shader_program);

    FileDescription* compileBinaryBlob(const std::string& filename, FileLoader* file_loader,
                                       const std::string* options = nullptr, uint32_t options_count = 0);
    void destroyRenderResource(FileDescription* blob);
};
}

#endif // _GL_SHADER_COMPILER_HH_