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
#include "tempest/graphics/rendering-definitions.hh"

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
    // , FetchShader <-- It is something that exists. However, have some doubts that it is going to persist.
    //                   Probably going to be replaced by direct vertex pulling.
};

enum class ElementType
{
    Scalar,
    Vector,
    Matrix,
    Array,
    Sampler,
    Struct,
    Shader,
    Profile,
    CompiledShader
};

enum class BufferType
{
    Instance
};

class PassShaderDescription
{
    size_t                  m_ShaderIndex;
    string                  m_AdditionalOpts;
public:
    PassShaderDescription(size_t shader_index, string additional_opts);
     ~PassShaderDescription();

    size_t getShaderIndex() const { return m_ShaderIndex; }
    string getAdditionalOptions() const { return m_AdditionalOpts; }
};

typedef std::vector<PassShaderDescription> PassShaderVector;

class PassDescription
{
    string                  m_Name;
    PassShaderVector        m_StageShaders;
public:
    PassDescription(string name)
        :   m_Name(name) {}

    void addShader(PassShaderDescription shader) { m_StageShaders.push_back(shader); }

    const PassShaderDescription& getAttachedShaderName(size_t idx) const { return m_StageShaders[idx]; }
    size_t getAttachedShaderCount() const { return m_StageShaders.size(); }
};

class InputParameter
{
    ElementType             m_Type;
    string                  m_Name;
    string                  m_Semantic;
public:
    InputParameter(ElementType _type, string name, string _semantic)
        :   m_Type(_type),
            m_Name(name),
            m_Semantic(_semantic) {}

    ElementType getType() const { return m_Type; }
    string getName() const { return m_Name; }
    string getSemantic() const { return m_Semantic; }
};

typedef std::vector<PassDescription>      PassVector;

class TechniqueDescription
{
    string                  m_Name;
    PassVector              m_Passes;
public:
    TechniqueDescription(string name)
        :   m_Name(name) {}

    string getName() const { return m_Name; }

    size_t getPassCount() const { return m_Passes.size(); }
    const PassDescription& getPass(size_t idx) const { return m_Passes[idx]; }
    void addPass(PassDescription pass) { m_Passes.push_back(pass); }
};

class ParameterDescription
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

typedef std::vector<ParameterDescription> ParameterVector;

struct SamplerDescription
{
    string                    m_Name;
    ParameterVector           m_Parameters;
public:
    SamplerDescription(string name)
        :   m_Name(name) {}

    void addParameter(ParameterDescription param) { m_Parameters.push_back(param); }

    string getName() const { return m_Name; }
    const ParameterDescription& getParameter(size_t idx) const { return m_Parameters[idx]; }
    size_t getParameterCount() const { return m_Parameters.size(); }
};

class BufferElement
{
    UniformValueType        m_Type;
    string                  m_Name;
    size_t                  m_ElementCount;
public:
    BufferElement(UniformValueType _type, string name, size_t elem_count)
        :   m_Type(_type),
            m_Name(name),
            m_ElementCount(elem_count) {}
    
    string getElementName() const { return m_Name; }
    UniformValueType getElementType() const { return m_Type; }
    size_t getElementCount() const { return m_ElementCount; }
};

typedef std::vector<BufferElement>        BufferElementVector;

class BufferDescription
{
    BufferType             m_BufferType;
    string                 m_Name;
    BufferElementVector    m_Elements;
public:
    BufferDescription(BufferType buffer_type, string name)
        :   m_BufferType(buffer_type),
            m_Name(name) {}
    
    void addBufferElement(BufferElement elem) { m_Elements.push_back(elem); }
    
    string getBufferName() const { return m_Name; }
    const BufferElement& getElement(size_t idx) const { return m_Elements[idx]; }
    size_t getElementCount() const { return m_Elements.size(); }
    BufferType getBufferType() const { return m_BufferType; }
};

typedef std::vector<InputParameter>       InputParameterVector;

class ShaderDescription
{
    ShaderType              m_ShaderType;
    string                  m_Name;
    string                  m_Content;
    InputParameterVector    m_InputSignature;
public:
    ShaderDescription(ShaderType shader_type, string name)
        :   m_ShaderType(shader_type),
            m_Name(name) {}

    void addInputParameter(InputParameter param) { m_InputSignature.push_back(param); }
    void appendContent(string content) { m_Content += content; }

    ShaderType getShaderType() const { return m_ShaderType; }
    string getName() const { return m_Name; }
    string getContent() const { return m_Content; }

    size_t getInputParameterCount() const { return m_InputSignature.size(); }
    const InputParameter& getInputParameter(size_t idx) const { return m_InputSignature[idx]; }
};

typedef std::vector<BufferDescription>    BufferVector;
typedef std::vector<SamplerDescription>   SamplerVector;
typedef std::vector<ShaderDescription>    ShaderVector;
typedef std::vector<TechniqueDescription> TechniqueVector;
typedef std::vector<string>               ImportedVector;

class EffectDescription
{
    ShaderVector            m_Shaders;
    SamplerVector           m_Samplers;
    TechniqueVector         m_Techniques;
    ImportedVector          m_Imported;
    BufferVector            m_Buffers;
public:
    EffectDescription() {}
     ~EffectDescription() {}

    void clear();

    void addBuffer(BufferDescription buffer) { m_Buffers.push_back(buffer); }
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
    const BufferDescription& getBuffer(size_t idx) const { return m_Buffers[idx]; }
    size_t getBufferCount() const { return m_Buffers.size(); }
};
}
}

#endif // _TEMPEST_EFFECT_COMMON_HH_
