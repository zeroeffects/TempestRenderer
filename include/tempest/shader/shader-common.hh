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
#include <limits>

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
    ComputeShader,
    // , FetchShader <-- It is something that exists. However, have some doubts that it is going to persist.
    //                   Probably going to be replaced by direct vertex pulling.
    ShaderTypeCount
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
    Regular,     //!< Generally this is buffer in memory. It comes with the expected latency and bandwidth limitations.
    Constant,    /*!< On some graphics cards its content might be assigned to registers. */
    Resource,    //!< This one is mostly for non-bindless style APIs that declare stuff in weird fashion.
    StructBuffer //!< There is another way to specify it. However, still available to avoid enum duplication.
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

    string getName() const { return m_Name; }
    
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

class BufferElement
{
    UniformValueType        m_Type;
    string                  m_Name;
    size_t                  m_BufferOffset;
    size_t                  m_ELementSize;
    size_t                  m_ElementCount;
public:
    BufferElement(size_t offset, UniformValueType _type, string name, size_t elem_size, size_t elem_count)
        :   m_BufferOffset(offset),
            m_Type(_type),
            m_Name(name),
            m_ELementSize(elem_size),
            m_ElementCount(elem_count) {}
    
    size_t getBufferOffset() const { return m_BufferOffset; }
    string getElementName() const { return m_Name; }
    UniformValueType getElementType() const { return m_Type; }
    size_t getElementSize() const { return m_ELementSize; }
    size_t getElementCount() const { return m_ElementCount; }
};

typedef std::vector<BufferElement>        BufferElementVector;

class BufferDescription
{
    BufferType             m_BufferType;
    string                 m_Name;
    size_t                 m_ResizablePart;
    BufferElementVector    m_Elements;
public:
    BufferDescription(BufferType buffer_type, string name)
        :   m_BufferType(buffer_type),
            m_Name(name),
            m_ResizablePart(std::numeric_limits<size_t>::max()) {}
    
    void addBufferElement(BufferElement elem) { m_Elements.push_back(elem); }
    
    void setResizablePart(size_t size) { TGE_ASSERT(m_ResizablePart == std::numeric_limits<size_t>::max(),
                                                    "More than one resizable part is not supported because it creates ambiguity.");
                                        m_ResizablePart = size; }
    size_t getResiablePart() const { return m_ResizablePart; }
    
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
typedef std::vector<ShaderDescription>    ShaderVector;
typedef std::vector<TechniqueDescription> TechniqueVector;
typedef std::vector<string>               ImportedVector;

class EffectDescription
{
    ShaderVector             m_Shaders;
    TechniqueVector          m_Techniques;
    ImportedVector           m_Imported;
    BufferVector             m_Buffers;
public:
    EffectDescription() {}
     ~EffectDescription() {}

    void clear();

    void addShader(ShaderDescription shader) { m_Shaders.push_back(shader); }
    void addTechnique(TechniqueDescription technique) { m_Techniques.push_back(technique); }
    void addImportedFile(string name) { m_Imported.push_back(name); }

    const ShaderDescription& getShader(size_t idx) const { return m_Shaders[idx]; }
    size_t getShaderCount() const { return m_Shaders.size(); }
    const TechniqueDescription& getTechnique(size_t idx) const { return m_Techniques[idx]; }
    size_t getTechniqueCount() const { return m_Techniques.size(); }
    string getImportedFile(size_t idx) const { return m_Imported[idx]; }
    size_t getImportedFileCount() const { return m_Imported.size(); }
    
    const BufferDescription& getBuffer(size_t idx) const { return m_Buffers[idx]; }
    size_t getBufferCount() const { return m_Buffers.size(); }
    void addBuffer(BufferDescription buffer) { m_Buffers.push_back(buffer); }
};
}
}

#endif // _TEMPEST_EFFECT_COMMON_HH_
