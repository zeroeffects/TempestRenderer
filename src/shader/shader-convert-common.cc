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

#include "tempest/shader/shader-convert-common.hh"
#include "tempest/shader/shader-ast.hh"

namespace Tempest
{
namespace Shader
{
static uint32 ConvertVariable(const string& base, const Shader::Variable* var, uint32* offset, Shader::BufferDescription* buf_desc);

static void ConvertType(const string& base, const Shader::Type* _type, UniformValueType* uniform_type, uint32* elem_size, uint32* offset, Shader::BufferDescription* buf_desc)
{
    switch(_type->getTypeEnum())
    {
    case Shader::ElementType::Vector:
    {
        auto* vector_type = _type->extract<Shader::VectorType>();
        auto* basic_type = vector_type->getBasicType()->extract<Shader::ScalarType>();
        switch(vector_type->getDimension())
        {
        case 2:
        {
            if(basic_type->getNodeName() == "int")
                *uniform_type = UniformValueType::IntegerVector2;
            else if(basic_type->getNodeName() == "bool")
                *uniform_type = UniformValueType::BooleanVector2;
            else if(basic_type->getNodeName() == "uint")
                *uniform_type = UniformValueType::UnsignedIntegerVector2;
            else if(basic_type->getNodeName() == "float")
                *uniform_type = UniformValueType::Vector2;
            else
                TGE_ASSERT(false, "Unknown vector type");
        } break;
        case 3:
        {
            if(basic_type->getNodeName() == "int")
                *uniform_type = UniformValueType::IntegerVector3;
            else if(basic_type->getNodeName() == "bool")
                *uniform_type = UniformValueType::BooleanVector3;
            else if(basic_type->getNodeName() == "uint")
                *uniform_type = UniformValueType::UnsignedIntegerVector3;
            else if(basic_type->getNodeName() == "float")
                *uniform_type = UniformValueType::Vector3;
            else
                TGE_ASSERT(false, "Unknown vector type");
        } break;
        case 4:
        {
            if(basic_type->getNodeName() == "int")
                *uniform_type = UniformValueType::IntegerVector4;
            else if(basic_type->getNodeName() == "bool")
                *uniform_type = UniformValueType::BooleanVector4;
            else if(basic_type->getNodeName() == "uint")
                *uniform_type = UniformValueType::UnsignedIntegerVector4;
            else if(basic_type->getNodeName() == "float")
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
        for(auto elem_iter = struct_body->current(), elem_iter_end = struct_body->end(); elem_iter != elem_iter_end; ++elem_iter)
        {
            TGE_ASSERT(elem_iter->getNodeType() == Shader::TGE_EFFECT_DECLARATION, "Expecting only declarations");
            auto* decl = elem_iter->extract<Shader::Declaration>();
            auto* vars = decl->getVariables();
            switch(vars->getNodeType())
            {
            case Shader::TGE_EFFECT_VARIABLE:
            {
                ConvertVariable(base, vars->extract<Shader::Variable>(), &struct_offset, buf_desc);
            } break;
            case Shader::TGE_AST_LIST_ELEMENT:
            {
                auto _list = vars->extract<AST::ListElement>();
                for(auto var_iter = _list->current(), var_iter_end = _list->end(); var_iter != var_iter_end; ++var_iter)
                {
                    TGE_ASSERT(var_iter->getNodeType() == Shader::TGE_EFFECT_VARIABLE, "Expecting variable");
                    auto* var = var_iter->extract<Shader::Variable>();
                    ConvertVariable(base, var, &struct_offset, buf_desc);
                }
            } break;
            default: TGE_ASSERT(false, "Unexpected node");
            }
        }
        *uniform_type = UniformValueType::Struct;
        *elem_size = struct_offset;
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
    case UniformValueType::Texture: return sizeof(uint64);
    case UniformValueType::Struct: return 4 * sizeof(float);
    default: TGE_ASSERT(false, "Unexpected uniform type");
    }
    return 0;
}

static uint32 ConvertVariable(const string& base, const Shader::Variable* var, uint32* offset, Shader::BufferDescription* buf_desc)
{
    UniformValueType uniform_type;
    uint32           elem_size,
                     array_size = 1;
    auto*            _type = var->getType();
    auto             type_enum = _type->getTypeEnum();
    string           var_name = var->getNodeName();

    switch(type_enum)
    {
    case Shader::ElementType::Struct:
    case Shader::ElementType::Vector:
    case Shader::ElementType::Matrix:
    {
         ConvertType(base.empty() ? var_name : (base + "." + var_name), _type, &uniform_type, &elem_size, offset, buf_desc);
    } break;
    case Shader::ElementType::Array:
    {
        auto* array_type = _type->extract<Shader::ArrayType>();
        ConvertType(base.empty() ? var_name : (base + "." + var_name), array_type->getBasicType(), &uniform_type, &elem_size, offset, buf_desc);
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
        elem_size = sizeof(uint64);
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

void ConvertBuffer(const Buffer* buffer, Shader::EffectDescription* fx_desc)
{
    uint32 offset = 0;
    Shader::BufferDescription buf_desc(buffer->getBufferType(), buffer->getNodeName());
    auto* list = buffer->getBody();
    for(auto iter = list->current(), iter_end = list->end(); iter != iter_end; ++iter)
    {
        auto* decl = iter->extract<Shader::Declaration>();
        auto* var = decl->getVariables()->extract<Shader::Variable>();

        auto _type = var->getType();
        ConvertVariable("", var, &offset, &buf_desc);
    }
    if(buf_desc.getResiablePart() == std::numeric_limits<uint32>::max())
        buf_desc.setResizablePart(0);
    fx_desc->addBuffer(buf_desc);
}

uint32 ConvertStructBuffer(const Variable* var, Shader::EffectDescription* fx_desc)
{
    TGE_ASSERT(var->getStorage() == Shader::StorageQualifier::StructBuffer, "Input variable should be of StructBuffer type");
    uint32 offset = 0;
    Shader::BufferDescription buf_desc(BufferType::StructBuffer, var->getNodeName());
    uint32 elem_size = ConvertVariable("", var, &offset, &buf_desc);
    fx_desc->addBuffer(buf_desc);
    return elem_size;
}
}
}