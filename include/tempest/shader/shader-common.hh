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
#include <cstring>

namespace Tempest
{
namespace Shader
{
enum class ShaderType: uint32
{
    VertexShader,
    TessellationControlShader,
    TessellationEvaluationShader,
    GeometryShader,
    FragmentShader,
    ComputeShader,
    // , FetchShader <-- It is something that exists. However, have some doubts that it is going to persist.
    //                   Probably going to be replaced by direct vertex pulling.
    ShaderTypeCount,
    GenericShader = ShaderTypeCount
};

const char* ConvertShaderTypeToText(ShaderType _type);

enum class ElementType
{
    Scalar,
    Vector,
    Matrix,
    Array,
    Sampler,
    Struct
};

enum class BufferType
{
    Regular,     //!< Generally this is buffer in memory. It comes with the expected latency and bandwidth limitations.
    Constant,    /*!< On some graphics cards its content might be assigned to registers. */
    Resource,    //!< This one is mostly for non-bindless style APIs that declare stuff in weird fashion.
    StructBuffer //!< There is another way to specify it. However, still available to avoid enum duplication.
};

class BufferElement
{
    UniformValueType        m_Type;
    string                  m_Name;
    uint32                  m_BufferOffset;
    uint32                  m_ElementSize;
    uint32                  m_ElementCount;
public:
    BufferElement(uint32 offset, UniformValueType _type, string name, uint32 elem_size, uint32 elem_count)
        :   m_BufferOffset(offset),
            m_Type(_type),
            m_Name(name),
            m_ElementSize(elem_size),
            m_ElementCount(elem_count) {}
    
    uint32 getBufferOffset() const { return m_BufferOffset; }
    string getElementName() const { return m_Name; }
    UniformValueType getElementType() const { return m_Type; }
    uint32 getElementSize() const { return m_ElementSize; }
    uint32 getElementCount() const { return m_ElementCount; }
};

typedef std::vector<BufferElement>        BufferElementVector;

class BufferDescription
{
    BufferType             m_BufferType;
    string                 m_Name;
    uint32                 m_ResizablePart;
    BufferElementVector    m_Elements;
public:
    BufferDescription(BufferType buffer_type, string name)
        :   m_BufferType(buffer_type),
            m_Name(name),
            m_ResizablePart(std::numeric_limits<uint32>::max()) {}
    
    void addBufferElement(BufferElement elem) { m_Elements.push_back(elem); }
    
    void setResizablePart(uint32 size) { TGE_ASSERT(m_ResizablePart == std::numeric_limits<uint32>::max(),
                                                    "More than one resizable part is not supported because it creates ambiguity.");
                                        m_ResizablePart = size; }
    uint32 getResiablePart() const { return m_ResizablePart; }
    
    string getBufferName() const { return m_Name; }
    const BufferElement& getElement(uint32 idx) const { return m_Elements[idx]; }
    uint32 getElementCount() const { return static_cast<uint32>(m_Elements.size()); }
    BufferType getBufferType() const { return m_BufferType; }
};

class ShaderDescription
{
    string                  m_AdditionalOpts;
    string                  m_Content;
public:
    ShaderDescription() = default;

    void appendContent(string content) { m_Content += content; }

    string getContent() const { return m_Content; }
    void setAdditionalOptions(string opts) { m_AdditionalOpts = opts; }
    string getAdditionalOptions() const { return m_AdditionalOpts; }
};

struct VertexAttributeDescription
{
    uint32      BufferId = 0;
    string      Name; // For annoying validation purposes.
    DataFormat  Format = DataFormat::Unknown;
    uint32      Offset = 0;
    uint32      StepRate = 0;
};

typedef std::vector<BufferDescription>          BufferVector;
typedef std::vector<string>                     ImportedVector;
typedef std::vector<VertexAttributeDescription> VertexAttributeVector;

class EffectDescription
{
    ShaderDescription*       m_Shaders[(size_t)ShaderType::ShaderTypeCount];
    ImportedVector           m_Imported;
    BufferVector             m_Buffers;
    VertexAttributeVector    m_InputSignature;
public:
    EffectDescription() { memset(m_Shaders, 0, sizeof(m_Shaders)); }
     ~EffectDescription()
     {
         for(auto* shader : m_Shaders)
             delete shader;
     }

    EffectDescription(const EffectDescription& effect) = delete;
    EffectDescription& operator=(const EffectDescription& effect) = delete;

    void clear() { memset(m_Shaders, 0, sizeof(m_Shaders)); }

    void addImportedFile(string name) { m_Imported.push_back(name); }

    void addVertexAttribute(VertexAttributeDescription desc) { m_InputSignature.push_back(desc); }

    bool trySetShader(ShaderType _type, ShaderDescription* shader_desc)
    {
        auto& shader = m_Shaders[static_cast<size_t>(_type)];
        if(shader)
            return false;
        shader = shader_desc;
        return true;
    }
    const ShaderDescription* getShader(ShaderType _type) const { return m_Shaders[static_cast<size_t>(_type)]; }
    string getImportedFile(uint32 idx) const { return m_Imported[idx]; }
    uint32 getImportedFileCount() const { return static_cast<uint32>(m_Imported.size()); }
    
    const VertexAttributeDescription& getVertexAttribute(uint32 idx) const { return m_InputSignature[idx]; }
    uint32 getVertexAttributeCount() const { return m_InputSignature.size(); }

    const BufferDescription& getBuffer(uint32 idx) const { return m_Buffers[idx]; }
    uint32 getBufferCount() const { return static_cast<uint32>(m_Buffers.size()); }
    void addBuffer(BufferDescription buffer) { m_Buffers.push_back(buffer); }
};
}
}

#endif // _TEMPEST_EFFECT_COMMON_HH_
