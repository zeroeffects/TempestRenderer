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

#ifndef _TEMPEST_SHADER_HH_
#define _TEMPEST_SHADER_HH_

#include "tempest/utils/patterns.hh"
#include "tempest/graphics/rendering-definitions.hh"

namespace Tempest
{
struct DataDescription
{
    std::string      Name;
    UniformValueType Type;
    uint16_t         ElementSize;
    uint16_t         ElementCount;
    uint32_t         Offset;
};

struct ResourceTableDescription
{
    std::string                    Name;
    uint32_t                       BindPoint;
    uint32_t                       BufferSize;
    uint32_t                       ExtendablePart;
    PACKED_DATA(DataDescription) Uniforms;

    ResourceTableDescription(const ResourceTableDescription&) = delete;
    ResourceTableDescription& operator=(const ResourceTableDescription&) = delete;

private:
    ResourceTableDescription(uint32_t count, uint32_t extendable_part,std::string name, uint32_t bind_point)
        :   ExtendablePart(extendable_part),
            Name(name),
            BindPoint(bind_point),
            Uniforms(count) {}
    ~ResourceTableDescription() = default;
};

struct ResourceIndex
{
    uint32_t ResourceTableIndex = 0;
    uint32_t BaseOffset = 0;
};

inline bool operator==(const ResourceIndex& lhs, const ResourceIndex& rhs)
{
    return lhs.ResourceTableIndex == rhs.ResourceTableIndex &&
           lhs.BaseOffset == rhs.BaseOffset;
}

class BakedResourceTable
{
    char*                   m_Table;
    size_t                  m_Size;
public:
    BakedResourceTable()
        :   m_Table(nullptr),
            m_Size(0) {}

    BakedResourceTable(size_t size)
        :   m_Table(new char[size]),
            m_Size(size) {}
    ~BakedResourceTable() { delete[] m_Table; }

    void realloc(size_t size)
    {
        delete[] m_Table;
        m_Table = new char[size];
        m_Size = size;
    }

    void reset()
    {
        if(m_Table == nullptr)
            m_Table = new char[m_Size];
    }

    BakedResourceTable(BakedResourceTable&& table)
    {
        m_Table = table.m_Table;
        m_Size = table.m_Size;
        table.m_Table = nullptr;
    }

    BakedResourceTable& operator=(BakedResourceTable&& table)
    {
        m_Table = table.m_Table;
        m_Size = table.m_Size;
        table.m_Table = nullptr;
        return *this;
    }

    template<class T>
    void setValue(size_t offset, const T& val)
    {
        *reinterpret_cast<T*>(m_Table + offset) = val;
    }

    operator bool() const { return m_Table != nullptr; }

    const char* get() const { return m_Table; }
    char* get() { return m_Table; }

    size_t getSize() const { return m_Table ? m_Size : 0; }
};

const uint32_t InvalidResourceIndex = std::numeric_limits<uint32_t>::max();

class Matrix4;
union Vector4;
union Vector3;
union Vector2;

template<class T> struct UniformValueBinding;
#define UNIFORM_VALUE_BINDING(type, value) \
    template<> struct UniformValueBinding<type> { \
        static const UniformValueType value_type = value; };
UNIFORM_VALUE_BINDING(Matrix4, UniformValueType::Matrix4);
UNIFORM_VALUE_BINDING(Vector4, UniformValueType::Vector4);
UNIFORM_VALUE_BINDING(Vector3, UniformValueType::Vector3);
UNIFORM_VALUE_BINDING(Vector2, UniformValueType::Vector2);
UNIFORM_VALUE_BINDING(float, UniformValueType::Float);
UNIFORM_VALUE_BINDING(int32_t, UniformValueType::Integer);
UNIFORM_VALUE_BINDING(uint32_t, UniformValueType::UnsignedInteger);
UNIFORM_VALUE_BINDING(bool, UniformValueType::Boolean);
}

#endif // _TEMPEST_SHADER_HH_