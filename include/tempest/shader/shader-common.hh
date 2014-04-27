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

#ifndef _TEMPEST_EFFECT_COMMON_HH_
#define _TEMPEST_EFFECT_COMMON_HH_

#include "tempest/utils/types.hh"

#include <vector>

namespace Tempest
{
namespace Shader
{
enum class ShaderType
{
    GenericShader,
    VertexShader,
    TessellationControlShader,
    TessellationEvaluationShader,
    GeometryShader,
    FragmentShader,
    ComputeShader
    // TGE_EFFECT_FETCH_SHADER // <-- I bet that Mantle is going to have one
};

enum TypeEnum
{
    TGE_EFFECT_SCALAR_TYPE,
    TGE_EFFECT_VECTOR_TYPE,
    TGE_EFFECT_MATRIX_TYPE,
    TGE_EFFECT_ARRAY_TYPE,
    TGE_EFFECT_SAMPLER_TYPE,
    TGE_EFFECT_STRUCT_TYPE,
    TGE_EFFECT_SHADER_TYPE,
    TGE_EFFECT_PROFILE_TYPE,
    TGE_EFFECT_COMPILED_SHADER_TYPE
};

class PassShaderDescription
{
    string                  m_Name;
    string                  m_AdditionalOpts;
public:
    PassShaderDescription(string name, string additional_opts);
     ~PassShaderDescription();

    string getName() const { return m_Name; }
    string getAdditionalOptions() const { return m_AdditionalOpts; }
};

class PassDescription
{
    string                              m_Name;
    std::vector<PassShaderDescription>  m_StageShaders;
public:
    PassDescription(string name)
        :   m_Name(name) {}

    void addShader(PassShaderDescription shader) { m_StageShaders.push_back(shader); }

    const PassShaderDescription& getAttachedShaderName(size_t idx) const { return m_StageShaders[idx]; }
    size_t getAttachedShaderCount() const { return m_StageShaders.size(); }
};

struct InputParameter
{
    TypeEnum                m_Type;
    string                  m_Name;
    string                  m_Semantic;
public:
    InputParameter(TypeEnum _type, string name, string _semantic);

    TypeEnum getType() const { return m_Type; }
    string getName() const { return m_Name; }
    string getSemantic() const { return m_Semantic; }
};

class TechniqueDescription
{
    typedef std::vector<PassDescription> PassList;
    string                  m_Name;
    PassList                m_Passes;
public:
    TechniqueDescription(string name)
        :   m_Name(name) {}

    string getName() const { return m_Name; }

    size_t getPassCount() const { return m_Passes.size(); }
    const PassDescription& getPass(size_t idx) const { return m_Passes[idx]; }
    void addPass(PassDescription pass) { m_Passes.push_back(pass); }
};

struct ParameterDescription
{
    string                  m_Name;
    string                  m_Value;
public:
    ParameterDescription(string name, string value)
        :   m_Name(name),
            m_Value(value) {}

    string getName() const { return m_Name; }
    string getValue() const { return m_Value; }
};
struct SamplerDescription
{
    typedef std::vector<ParameterDescription> ParameterList;
    string                  m_Name;
    ParameterList           m_Parameters;
public:
    SamplerDescription(string name)
        :   m_Name(name) {}

    void addParameter(ParameterDescription param) { m_Parameters.push_back(param); }

    string getName() const { return m_Name; }
    const ParameterDescription& getParameter(size_t idx) const { return m_Parameters[idx]; }
    size_t getParameterCount() const { return m_Parameters.size(); }
};

struct ShaderDescription
{
    typedef std::vector<InputParameter> SamplerList;
    typedef std::vector<InputParameter> InputParameterList;
    ShaderType              m_ShaderType;
    string                  m_Name;
    string                  m_Content;
    SamplerList             m_Samplers;
    InputParameterList      m_InputSignature;
public:
    ShaderDescription(ShaderType shader_type, string name)
        :   m_ShaderType(shader_type),
            m_Name(name) {}

    void addInputParameter(InputParameter param) { m_InputSignature.push_back(param); }
    void addSampler(string name, string texture) { m_Samplers.push_back(InputParameter(TGE_EFFECT_SAMPLER_TYPE, name, texture)); }
    void appendContent(string content) { m_Content += content; }

    ShaderType getShaderType() const { return m_ShaderType; }
    string getName() const { return m_Name; }
    string getContent() const { return m_Content; }

    string getSamplerName(size_t idx) const { return m_Samplers[idx].getName(); }
    string getSamplerTexture(size_t idx) const { return m_Samplers[idx].getSemantic(); }
    size_t getSamplerCount() const { return m_Samplers.size(); }

    size_t getInputParameterCount() const { return m_InputSignature.size(); }
    const InputParameter& getInputParameter(size_t idx) const { return m_InputSignature[idx]; }
};

class EffectDescription
{
    typedef std::vector<ShaderDescription>    ShaderList;
    typedef std::vector<TechniqueDescription> TechniqueList;
    typedef std::vector<SamplerDescription>   SamplerList;
    typedef std::vector<string>               ImportedList;
    ShaderList              m_Shaders;
    SamplerList             m_Samplers;
    TechniqueList           m_Techniques;
    ImportedList            m_Imported;
public:
    EffectDescription() {}
     ~EffectDescription() {}

    void clear();

    void addShader(ShaderDescription shader) { m_Shaders.push_back(shader); }
    void addSampler(SamplerDescription sampler) { m_Samplers.push_back(sampler); }
    void addTechnique(TechniqueDescription technique) { m_Techniques.push_back(technique); }
    void addImportedFile(string name) { m_Imported.push_back(name); }

    const ShaderDescription& getShader(size_t idx) const { return m_Shaders[idx]; }
    size_t getShaderCount() const { return m_Shaders.size(); }
    const TechniqueDescription& getTechnique(size_t idx) const { return m_Techniques[idx]; }
    size_t getTechniqueCount() const { return m_Techniques.size(); }
    const SamplerDescription& getSampler(size_t idx) const { return m_Samplers[idx]; }
    size_t getSamplerCount() const { return m_Samplers.size(); }
    string getImportedFile(size_t idx) const { return m_Imported[idx]; }
    size_t getImportedFileCount() const { return m_Imported.size(); }

};
}
}

#endif // _TEMPEST_EFFECT_COMMON_HH_
