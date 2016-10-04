/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2015 Zdravko Velinov
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

#include "tempest/mesh/obj-mtl-parser.hh"
#include "tempest/utils/assert.hh"

YY_DECL;

namespace Tempest
{
namespace ObjMtlLoader
{
static bool IsNumberToken(ObjMtlToken token)
{
    return token == ObjMtlToken::Integer || token == ObjMtlToken::Float;
}    

static bool IsEndToken(ObjMtlToken token)
{
    return token == ObjMtlToken::EndOfFile || token == ObjMtlToken::EndOfLine;
}

const char* TranslateToken(ObjMtlToken token)
{
    switch(token)
    {
    #define OBJ_MTL_TOKEN(token_enum, token_name) case ObjMtlToken::token_enum: return token_name;
    #include "tempest/mesh/obj-mtl-tokens.hh"
    #undef OBJ_MTL_TOKEN
    default: TGE_ASSERT(false, "Unknown token"); return "";
    }
}

Parser::Parser(ObjMtlLoader::Driver& driver)
    :   m_Driver(driver)
{
    m_CurrentLocation.filename = &driver.__FileName;
}

bool Parser::parseNumber(float* result, ObjMtlToken* res_token)
{
    SemanticType semantic;
    *res_token = ObjMtlLoaderLexer(&semantic, &m_CurrentLocation, m_Driver);
    switch(*res_token)
    {
    case ObjMtlToken::Float:
    {
        *result = semantic.FloatValue;
    } break;
    case ObjMtlToken::Integer:
    {
        *result = static_cast<float>(semantic.IntegerValue);
    } break;
    default:
    {
        std::stringstream ss;
        ss << "Unexpected \"" << TranslateToken(*res_token) << "\". Expecting number instead.";
        m_Driver.error(m_CurrentLocation, ss.str());
        return false;
    } break;
    }
    return true;
}

bool Parser::parseString(SemanticType* semantic, const char** result, ObjMtlToken* res_token)
{
    *res_token = ObjMtlLoaderLexer(semantic, &m_CurrentLocation, m_Driver);
    if(*res_token != ObjMtlToken::String)
    {
        std::stringstream ss;
        ss << "Unexpected \"" << TranslateToken(*res_token) << "\". Expecting string instead.";
        m_Driver.error(m_CurrentLocation, ss.str());
        return false;
    }
    *result = semantic->StringValue;
    return true;
}

static bool ExpectEnd(ObjMtlLoader::Driver& driver, const Location& loc, ObjMtlToken token)
{
    if(token == ObjMtlToken::EndOfFile || token == ObjMtlToken::EndOfLine)
        return true;
    std::stringstream ss;
    ss << "Unexpected " << TranslateToken(token) << ". Expecting end of current statement.";
    driver.error(loc, ss.str());
    return false;
}

void Parser::skipToEndOfLine(ObjMtlToken cur_token)
{
    SemanticType semantic;
    while(!IsEndToken(cur_token))
    {
        cur_token = ObjMtlLoaderLexer(&semantic, &m_CurrentLocation, m_Driver);
    }
}

#define SET_MATERIAL_FIELD(log_line, name, value) \
    if(m_Driver.getCurrentMaterial()) { \
        m_Driver.getCurrentMaterial()->name = value; \
    } else { \
        m_Driver.error(log_line, "Failed to initialize material field \"" #name "\" because no material was specified."); \
    }

#define PARSE_MATERIAL_FIELD_V3(log_line, name) \
    case ObjMtlToken::name: { \
        float x, y, z; \
        auto status = parseNumber(&x, &token) && parseNumber(&y, &token) && parseNumber(&z, &token) && ExpectEnd(m_Driver, log_line, ObjMtlLoaderLexer(&semantic, &log_line, m_Driver)); \
        if(!status) { \
            skipToEndOfLine(token); \
            break; \
        } \
        Vector3 vec = Tempest::Vector3{x, y, z}; \
        SET_MATERIAL_FIELD(log_line, name, vec); \
    } break;

#define PARSE_MATERIAL_FIELD_FLOAT(log_line, name) \
    case ObjMtlToken::name: { \
        float x; \
        auto status = parseNumber(&x, &token) && ExpectEnd(m_Driver, log_line, ObjMtlLoaderLexer(&semantic, &log_line, m_Driver)); \
        if(!status) { \
            skipToEndOfLine(token); \
            break; \
        } \
        SET_MATERIAL_FIELD(log_line, name, x); \
    } break;

#define PARSE_MATERIAL_FIELD_STRING(log_line, name) \
    case ObjMtlToken::name: { \
        const char* str; \
        auto status = parseString(&semantic, &str, &token) && ExpectEnd(m_Driver, log_line, ObjMtlLoaderLexer(&semantic, &log_line, m_Driver)); \
        if(!status) { \
            skipToEndOfLine(token); \
            break; \
        } \
        SET_MATERIAL_FIELD(log_line, name, str); \
    } break;
    
int Parser::parse()
{
    ObjMtlToken token;
    SemanticType semantic;
    while((token = ObjMtlLoaderLexer(&semantic, &m_CurrentLocation, m_Driver)) != ObjMtlToken::EndOfFile)
    {
        switch(token)
        {
        default:
        {
            std::stringstream ss;
            ss << "Unexpected \"", TranslateToken(token), "\". It is not a valid top level token or it is currently unsupported.";
            m_Driver.error(m_CurrentLocation, ss.str());
        } break;
        case ObjMtlToken::EndOfLine:
        {
            // ignore
        } break;
        case ObjMtlToken::NewMtl:
        {
            const char* name;
            auto status = parseString(&semantic, &name, &token);
            
            if(!status)
            {
                skipToEndOfLine(token);
                break;
            }
            
            m_Driver.pushNewMaterial(name);
        } break;
        PARSE_MATERIAL_FIELD_V3(m_CurrentLocation, AmbientReflectivity)
        PARSE_MATERIAL_FIELD_V3(m_CurrentLocation, DiffuseReflectivity)
        PARSE_MATERIAL_FIELD_V3(m_CurrentLocation, SpecularReflectivity)
        PARSE_MATERIAL_FIELD_V3(m_CurrentLocation, Emission)
        PARSE_MATERIAL_FIELD_V3(m_CurrentLocation, TransmissionFilter)
        case ObjMtlToken::Illum:
        {
            token = ObjMtlLoaderLexer(&semantic, &m_CurrentLocation, m_Driver);
            switch(token)
            {
            case ObjMtlToken::Integer:
            case ObjMtlToken::IllumModel:
            {
				int num = semantic.IntegerValue;
                auto status = ExpectEnd(m_Driver, m_CurrentLocation, ObjMtlLoaderLexer(&semantic, &m_CurrentLocation, m_Driver));
                if(status)
                {
                    SET_MATERIAL_FIELD(m_CurrentLocation, IllumModel, static_cast<ObjMtlLoader::IlluminationModel>(num));
                }
            } break;
            default:
            {
                m_Driver.error(m_CurrentLocation, "Invalid illumination model declaration");
            } break;
            }
        } break;
        case ObjMtlToken::Dissolve: {
            float x;
            auto status = parseNumber(&x, &token);
            if(status)
            {
                token = ObjMtlLoaderLexer(&semantic, &m_CurrentLocation, m_Driver);
                if(IsNumberToken(token))
                {
                    token = ObjMtlLoaderLexer(&semantic, &m_CurrentLocation, m_Driver);
                }
                status = ExpectEnd(m_Driver, m_CurrentLocation, token);
            }
            
            if(!status)
            {
                skipToEndOfLine(token);
                break;
            }
            SET_MATERIAL_FIELD(m_CurrentLocation, Dissolve, x);
        } break;
        PARSE_MATERIAL_FIELD_FLOAT(m_CurrentLocation, SpecularExponent)
        PARSE_MATERIAL_FIELD_FLOAT(m_CurrentLocation, ReflectionSharpness)
        PARSE_MATERIAL_FIELD_FLOAT(m_CurrentLocation, RefractionIndex)
        PARSE_MATERIAL_FIELD_STRING(m_CurrentLocation, AmbientReflectivityMap)
        PARSE_MATERIAL_FIELD_STRING(m_CurrentLocation, DiffuseReflectivityMap)
        PARSE_MATERIAL_FIELD_STRING(m_CurrentLocation, SpecularReflectivityMap)
        PARSE_MATERIAL_FIELD_STRING(m_CurrentLocation, SpecularExponentMap)
        PARSE_MATERIAL_FIELD_STRING(m_CurrentLocation, DissolveMap)
        }
    }
	return 0;
}
}
}