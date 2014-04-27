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

#include "tempest/utils/parse-command-line.hh"
#include "tempest/shader/gl-shader-generator.hh"
#include "tempest/shader/dx-shader-generator.hh"
#include "tempest/shader/file-loader.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/system.hh"
#include "tempest/utils/logging.hh"

#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace Tempest;

template<class... TArgs>
void GenerateError(TArgs&&... args)
{
    auto exe_path = System::GetExecutablePath();
    Tempest::Log(LogLevel::Error, Path(exe_path).filename(), ": error: ", args...);
}

// TODO: Replace include loaders
class DummyIncludeLoader: public FileLoader
{
public:
    DummyIncludeLoader() {}
    virtual ~DummyIncludeLoader() {}
    
    virtual FileDescription* loadFileContent(const string& name) override { return nullptr; }
    virtual void freeFileContent(FileDescription* ptr) override {}
};

enum CompilerFlags
{
    CompileToGLSL    = 0,
    CompileToHLSL    = 1,
    CompileFlagMask  = CompileToGLSL|CompileToHLSL,
    CompileFlagShift = 0
};

bool BuildTextShaderSimple(const string& input_file, std::ostream& output_file, uint32 flags)
{
    DummyIncludeLoader include_loader;
    Shader::EffectDescription effect;
    int compile_flag = (flags >> CompileFlagShift) & CompileFlagMask;
    if(compile_flag == CompileToGLSL)
    {
        if(!GLFX::LoadEffect(input_file, &include_loader, effect))
        {
            return false;
        }
    }
    else if(compile_flag == CompileToHLSL)
    {
        if(!DXFX::LoadEffect(input_file, &include_loader, effect))
        {
            return false;
        }
    }
    else
    {
        GenerateError("Unsupported shading language.");
        return false;
    }

    for(size_t i = 0, iend = effect.getShaderCount(); i < iend; ++i)
    {
        auto& shader = effect.getShader(i);
        output_file << "// " << shader.getName() << "\n"
                    << shader.getContent() << std::endl;
    }
}

bool BuildTextShaderSimple(const string& input_file, const string& output_file, uint32 flags)
{
    std::fstream fs(output_file.c_str(), std::ios::out);
    return BuildTextShaderSimple(input_file, output_file.empty() ? std::cout : fs, flags);
}

int TempestMain(int argc, char* argv[])
{
    CommandLineOptsParser parser("shader_compiler", true);
    
    parser.createOption('o', "output", "The name of the file to which the processed shader is going to be written.", true);
    parser.createOption('l', "language", "The shading language of the output file.", true);
    parser.parse(argc, argv);
    string output_file = parser.extract<string>("output");
    string shading_lang = parser.extract<string>("language");
    uint32 flags = 0;
    if(shading_lang.empty() || shading_lang == "glsl")
    {
        flags |= (CompileToGLSL << CompileFlagShift);
    }
    else if(shading_lang == "hlsl")
    {
        flags |= (CompileToHLSL << CompileFlagShift);
    }
    else
    {
        GenerateError("Unknown shading language: ", shading_lang);
        return EXIT_FAILURE;
    }
    
    if(output_file == "-")
    {
        output_file.clear();
    }
    if(parser.getUnassociatedCount() != 1)
    {
        GenerateError("Expecting input file");
        return EXIT_FAILURE;
    }
    string input_file = parser.getUnassociatedArgument(0);

    return BuildTextShaderSimple(input_file, output_file, flags) ? EXIT_SUCCESS : EXIT_FAILURE;
}
