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

#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/shader/shader-convert-common.hh"
#include "tempest/shader/shader-ast.hh"

namespace Tempest
{
namespace Shader
{
static uint32 ConvertVariable(const string* opts, size_t opts_count, const string& base, const Shader::Variable* var, uint32* offset, Shader::BufferDescription* buf_desc);
static void RecursiveConvertBuffer(const string* opts, size_t opts_count, const string& base, const List* list, uint32* offset, Shader::BufferDescription* buf_desc);

static void ConvertType(const string* opts, size_t opts_count, const string& base, const Shader::Type* _type, UniformValueType* uniform_type, uint32* elem_size, uint32* offset, Shader::BufferDescription* buf_desc)
{
    switch(_type->getTypeEnum())
    {
    case Shader::ElementType::Scalar:
    {
        auto basic_type_str = _type->getNodeName();
        if(basic_type_str == "int")
            *uniform_type = UniformValueType::Integer;
        else if(basic_type_str == "bool")
            *uniform_type = UniformValueType::Boolean;
        else if(basic_type_str == "uint")
            *uniform_type = UniformValueType::UnsignedInteger;
        else if(basic_type_str == "float")
            *uniform_type = UniformValueType::Float;
        else
            TGE_ASSERT(false, "Unknown vector type");
        
        *elem_size = UniformValueTypeSize(*uniform_type);
    } break;
    case Shader::ElementType::Vector:
    {
        auto* vector_type = _type->extract<Shader::VectorType>();
        auto* basic_type = vector_type->getBasicType()->extract<Shader::ScalarType>();
        auto basic_type_str = basic_type->getNodeName();
        switch(vector_type->getDimension())
        {
        case 2:
        {
            if(basic_type_str == "int")
                *uniform_type = UniformValueType::IntegerVector2;
            else if(basic_type_str == "bool")
                *uniform_type = UniformValueType::BooleanVector2;
            else if(basic_type_str == "uint")
                *uniform_type = UniformValueType::UnsignedIntegerVector2;
            else if(basic_type_str == "float")
                *uniform_type = UniformValueType::Vector2;
            else
                TGE_ASSERT(false, "Unknown vector type");
        } break;
        case 3:
        {
            if(basic_type_str == "int")
                *uniform_type = UniformValueType::IntegerVector3;
            else if(basic_type_str == "bool")
                *uniform_type = UniformValueType::BooleanVector3;
            else if(basic_type_str == "uint")
                *uniform_type = UniformValueType::UnsignedIntegerVector3;
            else if(basic_type_str == "float")
                *uniform_type = UniformValueType::Vector3;
            else
                TGE_ASSERT(false, "Unknown vector type");
        } break;
        case 4:
        {
            if(basic_type_str == "int")
                *uniform_type = UniformValueType::IntegerVector4;
            else if(basic_type_str == "bool")
                *uniform_type = UniformValueType::BooleanVector4;
            else if(basic_type_str == "uint")
                *uniform_type = UniformValueType::UnsignedIntegerVector4;
            else if(basic_type_str == "float")
                *uniform_type = UniformValueType::Vector4;
            else
                TGE_ASSERT(false, "Unknown vector type");
        } break;
        default:
            TGE_ASSERT(false, "Unknown vector type");
        }
        *elem_size = UniformValueTypeSize(*uniform_type);
    } break;
    case Shader::ElementType::Matrix:
    {
        auto* matrix_type = _type->extract<Shader::MatrixType>();
        size_t rows = matrix_type->getRows();
        size_t columns = matrix_type->getColumns();
        switch(columns)
        {
        case 2:
            switch(rows)
            {
            case 2: *uniform_type = UniformValueType::Matrix2; break;
            case 3: *uniform_type = UniformValueType::Matrix2x3; break;
            case 4: *uniform_type = UniformValueType::Matrix2x4; break;
            default: TGE_ASSERT(false, "Unsupported matrix type"); break;
            }
            break;
        case 3:
            switch(rows)
            {
            case 2: *uniform_type = UniformValueType::Matrix3x2; break;
            case 3: *uniform_type = UniformValueType::Matrix3; break;
            case 4: *uniform_type = UniformValueType::Matrix3x4; break;
            default: TGE_ASSERT(false, "Unsupported matrix type"); break;
            }
            break;
        case 4:
            switch(rows)
            {
            case 2: *uniform_type = UniformValueType::Matrix4x2; break;
            case 3: *uniform_type = UniformValueType::Matrix4x3; break;
            case 4: *uniform_type = UniformValueType::Matrix4; break;
            default: TGE_ASSERT(false, "Unsupported matrix type"); break;
            }
            break;
        default: TGE_ASSERT(false, "Unsupported matrix type"); break;
        }
        *elem_size = UniformValueTypeSize(*uniform_type);
    } break;
    case Shader::ElementType::Struct:
    {
        *offset = (*offset + 4 * sizeof(float)-1) & ~(4 * sizeof(float)-1);
        uint32 struct_offset = 0; // members are in relative offset units
        auto* struct_type = _type->extract<Shader::StructType>();
        auto* struct_body = struct_type->getBody();
        RecursiveConvertBuffer(opts, opts_count, base, struct_body, &struct_offset, buf_desc);
        *uniform_type = UniformValueType::Struct;
        *elem_size = struct_offset;
    } break;
    case Shader::ElementType::Sampler:
    {
        auto* sampler_type = _type->extract<Shader::SamplerType>();
        *uniform_type = UniformValueType::Texture;
        *elem_size = UniformValueTypeSize(*uniform_type);
    } break;
    default: TGE_ASSERT(false, "Unexpected type"); break;
    }
    TGE_ASSERT(*elem_size > 0, "Element size should be greater than one. Otherwise, it is pointless to define it");
}

static uint32 GetAlignment(UniformValueType _type)
{
    switch(_type)
    {
    case UniformValueType::Float: return sizeof(float);
    case UniformValueType::Vector2: return 2 * sizeof(float);
    case UniformValueType::Vector3: return 4 * sizeof(float);
    case UniformValueType::Vector4: return 4 * sizeof(float);
    case UniformValueType::Integer: return sizeof(int32);
    case UniformValueType::IntegerVector2: return 2 * sizeof(int32);
    case UniformValueType::IntegerVector3: return 4 * sizeof(int32);
    case UniformValueType::IntegerVector4: return 4 * sizeof(int32);
    case UniformValueType::UnsignedInteger: return sizeof(uint32);
    case UniformValueType::UnsignedIntegerVector2: return 2 * sizeof(uint32);
    case UniformValueType::UnsignedIntegerVector3: return 4 * sizeof(uint32);
    case UniformValueType::UnsignedIntegerVector4: return 4 * sizeof(uint32);
    case UniformValueType::Boolean: return sizeof(uint32);
    case UniformValueType::BooleanVector2: return 2 * sizeof(uint32);
    case UniformValueType::BooleanVector3: return 4 * sizeof(uint32);
    case UniformValueType::BooleanVector4: return 4 * sizeof(uint32);
    case UniformValueType::Matrix2: return 2 * 2 * sizeof(float);
    case UniformValueType::Matrix3: return 3 * 3 * sizeof(float);
    case UniformValueType::Matrix4: return 4 * 4 * sizeof(float);
    case UniformValueType::Matrix2x3: return 2 * 4 * sizeof(float);
    case UniformValueType::Matrix2x4: return 2 * 4 * sizeof(float);
    case UniformValueType::Matrix3x2: return 3 * 2 * sizeof(float);
    case UniformValueType::Matrix3x4: return 3 * 4 * sizeof(float);
    case UniformValueType::Matrix4x2: return 4 * 2 * sizeof(float);
    case UniformValueType::Matrix4x3: return 4 * 4 * sizeof(float);
    case UniformValueType::Texture:
    {
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
        if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_TEXTURE_BINDLESS))
        {
            return sizeof(uint64);
        }
        else
#endif
        {
            return sizeof(uint32);
        }
    }
    case UniformValueType::Struct: return 4 * sizeof(float);
    default: TGE_ASSERT(false, "Unexpected uniform type");
    }
    return 0;
}

static uint32 ConvertVariable(const string* opts, size_t opts_count, const string& base, const Shader::Variable* var, uint32* offset, Shader::BufferDescription* buf_desc)
{
    UniformValueType uniform_type;
    uint32           elem_size,
                     array_size = 1;
    auto*            _type = var->getType();
    auto             type_enum = _type->getTypeEnum();
    string           var_name = var->getNodeName();

    switch(type_enum)
    {
    case Shader::ElementType::Scalar:
    case Shader::ElementType::Struct:
    case Shader::ElementType::Vector:
    case Shader::ElementType::Matrix:
    {
        ConvertType(opts, opts_count, base.empty() ? var_name : (base + "." + var_name), _type, &uniform_type, &elem_size, offset, buf_desc);
    } break;
    case Shader::ElementType::Array:
    {
        auto* array_type = _type->extract<Shader::ArrayType>();
        ConvertType(opts, opts_count, base.empty() ? var_name : (base + "." + var_name), array_type->getBasicType(), &uniform_type, &elem_size, offset, buf_desc);
        TGE_ASSERT(array_size == 1, "Arrays of arrays are unsupported");
        auto* size = array_type->getSize();
        if(*size)
        {
            array_size = size->extract<AST::Value<int>>()->getValue();
        }
        else
        {
            array_size = 0; // infinite
            buf_desc->setResizablePart(elem_size);
        }
        *offset = (*offset + 4 * sizeof(float)-1) & ~(4 * sizeof(float)-1);
    } break;
    case Shader::ElementType::Sampler:
    {
        auto* sampler_type = _type->extract<Shader::SamplerType>();
        uniform_type = UniformValueType::Texture;
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
        if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_TEXTURE_BINDLESS))
        {
            elem_size = sizeof(uint64);
        }
        else
#endif
        {
            elem_size = sizeof(uint32);
        }
        // TODO
    } break;
    default:
        TGE_ASSERT(false, "Unsupported type"); break;
    }

    auto alignment = GetAlignment(uniform_type);
    *offset = (*offset + alignment - 1) & ~(alignment - 1);

    buf_desc->addBufferElement(Shader::BufferElement(*offset, uniform_type, base.empty() ? var_name : (base + "." + var_name), elem_size, array_size));

    if(array_size > 1 || uniform_type == UniformValueType::Struct)
    {
        *offset += array_size*((elem_size + 4 * sizeof(float)-1) & ~(4 * sizeof(float)-1));
    }
    else
    {
        *offset += elem_size;
    }
    return elem_size;
}

static void RecursiveConvertBuffer(const string* opts, size_t opts_count, const string& base, const List* list, uint32* offset, Shader::BufferDescription* buf_desc)
{
    for(auto iter = list->current(), iter_end = list->end(); iter != iter_end; ++iter)
    {
        auto node_type = iter->getNodeType();
        const Declaration* decl = nullptr;
        if(node_type == TGE_EFFECT_OPTIONAL)
        {
            auto* _opt = iter->extract<Optional>();
            auto* opts_end = opts + opts_count;
            if(std::find(opts, opts_end, _opt->getNodeName()) == opts_end)
                continue;
            auto* content = _opt->getContent();
            if(content->getNodeType() == TGE_AST_BLOCK)
            {
                RecursiveConvertBuffer(opts, opts_count, base, content->extract<Block>()->getBody(), offset, buf_desc);
                continue;
            }
            else if(content->getNodeType() == TGE_EFFECT_DECLARATION)
            {
                decl = content->extract<Declaration>();
            }
            else
            {
                TGE_ASSERT(false, "Unexpected mode");
            }
        }
        else if(node_type == TGE_EFFECT_DECLARATION)
        {
            decl = iter->extract<Declaration>();
        }
        else
        {
            TGE_ASSERT(false, "Unexpected node type");
            continue;
        }

        auto* var = decl->getVariables()->extract<Shader::Variable>();

        auto _type = var->getType();
        ConvertVariable(opts, opts_count, base, var, offset, buf_desc);
    }
}

void ConvertBuffer(const string* opts, size_t opts_count, const Buffer* buffer, Shader::EffectDescription* fx_desc)
{
    uint32 offset = 0;
    Shader::BufferDescription buf_desc(buffer->getBufferType(), buffer->getNodeName());
    auto* _list = buffer->getBody();
    RecursiveConvertBuffer(opts, opts_count, "", _list, &offset, &buf_desc);
    if(buf_desc.getResiablePart() == std::numeric_limits<uint32>::max())
        buf_desc.setResizablePart(0);
    fx_desc->addBuffer(buf_desc);
}

uint32 ConvertStructBuffer(const string* opts, size_t opts_count, const Variable* var, Shader::EffectDescription* fx_desc)
{
    TGE_ASSERT(var->getStorage() == Shader::StorageQualifier::StructBuffer, "Input variable should be of StructBuffer type");
    uint32 offset = 0;
    Shader::BufferDescription buf_desc(BufferType::StructBuffer, var->getNodeName());
    uint32 elem_size = ConvertVariable(opts, opts_count, "", var, &offset, &buf_desc);
    fx_desc->addBuffer(buf_desc);
    return elem_size;
}
}
}