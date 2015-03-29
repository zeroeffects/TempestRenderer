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

#include "tempest/mesh/obj-loader-driver.hh"
#include "tempest/mesh/obj-loader-parser.hh"
#include "tempest/parser/driver-base.hh"

YY_DECL;

namespace Tempest
{
namespace ObjLoader
{
static bool IsNumberToken(ObjFileToken token)
{
    return token == ObjFileToken::Integer || token == ObjFileToken::Float;
}

static bool IsEndToken(ObjFileToken token)
{
    return token == ObjFileToken::EndOfFile || token == ObjFileToken::EndOfLine;
}

const char* TranslateToken(ObjFileToken token)
{
    switch(token)
    {
    #define OBJ_LOADER_TOKEN(token_enum, token_name) case ObjFileToken::token_enum: return token_name;
    #include "tempest/mesh/obj-loader-tokens.hh"
    #undef OBJ_LOADER_TOKEN
    default: TGE_ASSERT(false, "Unknown token"); return "";
    }
}

static void ConvertNumber(ObjFileToken token, SemanticType* semantic, float* num)
{
    switch(token)
    {
    case ObjFileToken::Float:
    {
        *num = semantic->FloatValue;
    } break;
    case ObjFileToken::Integer:
    {
        *num = static_cast<float>(semantic->IntegerValue);
    } break;
    default: TGE_ASSERT(false, "Unsupported type");
    }
}

static bool ExpectEnd(ObjLoader::Driver& driver, const Location& loc, ObjFileToken token)
{
    if(token == ObjFileToken::EndOfFile || token == ObjFileToken::EndOfLine)
        return true;
    std::stringstream ss;
    ss << "Unexpected " << TranslateToken(token) << ". Expecting end of current statement.";
    driver.error(loc, ss.str());
    return false;
}

Parser::Parser(ObjLoader::Driver& driver)
    :   m_Driver(driver) {}

void Parser::skipToEndOfLine(ObjFileToken cur_token)
{
    SemanticType semantic;
    Location location;
    while(!IsEndToken(cur_token))
    {
        cur_token = ObjectLoaderLexer(&semantic, &location, m_Driver);
    }
}

bool Parser::parseNumber(float* result, ObjFileToken* res_token)
{
    SemanticType semantic;
    Location location;
    *res_token = ObjectLoaderLexer(&semantic, &location, m_Driver);
    switch(*res_token)
    {
    case ObjFileToken::Float:
    {
        *result = semantic.FloatValue;
    } break;
    case ObjFileToken::Integer:
    {
        *result = static_cast<float>(semantic.IntegerValue);
    } break;
    default:
    {
        std::stringstream ss;
        ss << "Unexpected \"" << TranslateToken(*res_token) << "\". Expecting number instead.";
        m_Driver.error(location, ss.str());
        return false;
    } break;
    }
    return true;
}

bool Parser::parseString(SemanticType* semantic, const char** result, ObjFileToken* res_token)
{
    Location location;
    *res_token = ObjectLoaderLexer(semantic, &location, m_Driver);
    if(*res_token != ObjFileToken::String)
    {
        std::stringstream ss;
        ss << "Unexpected \"" << TranslateToken(*res_token) << "\". Expecting string instead.";
        m_Driver.error(location, ss.str());
        return false;
    }
    *result = semantic->StringValue;
    return true;
}

bool Parser::parseIndex(int* result, ObjFileToken* res_token)
{
    SemanticType semantic;
    Location location;
    *res_token = ObjectLoaderLexer(&semantic, &location, m_Driver);
    if(*res_token != ObjFileToken::Integer)
    {
        std::stringstream ss;
        ss << "Unexpected \"" << TranslateToken(*res_token) << "\". Expecting number instead.";
        m_Driver.error(location, ss.str());
        return false;
    }
    
    *result = semantic.IntegerValue;
    return true;
}

bool Parser::expect(ObjFileToken token, ObjFileToken* res_token)
{
    SemanticType semantic;
    Location location;
    *res_token = ObjectLoaderLexer(&semantic, &location, m_Driver);
    if(*res_token != token)
    {
        std::stringstream ss;
        ss << "Unexpected \"" << TranslateToken(*res_token) << "\". Expecting \"" << TranslateToken(token) << "\" instead.";
        m_Driver.error(location, ss.str());
        return false;
    }
    return true;
}

void Parser::parseIndices(const Location& declaration_location)
{
    SemanticType semantic;
    Location location;
    ObjFileToken token;

    int start_pos_ind, prev_pos_ind, cur_pos_ind,
        start_tc_ind, prev_tc_ind, cur_tc_ind,
        start_norm_ind, prev_norm_ind, cur_norm_ind;

    auto success = parseIndex(&start_pos_ind, &token);

    if(success)
    {
        auto token = ObjectLoaderLexer(&semantic, &location, m_Driver);

        switch(token)
        {
        case ObjFileToken::Integer:
        {
            prev_pos_ind = semantic.IntegerValue;
            success &= parseIndex(&prev_pos_ind, &token) && parseIndex(&cur_pos_ind, &token);
            if(success)
            {
                do
                {
                    m_Driver.pushPositionIndex(start_pos_ind);
                    m_Driver.pushPositionIndex(prev_pos_ind);
                    m_Driver.pushPositionIndex(cur_pos_ind);
                    prev_pos_ind = cur_pos_ind;
                } while(parseIndex(&cur_pos_ind, &token));
                success &= IsEndToken(token);
            }
        } break;
        case static_cast<ObjFileToken>('/'):
        {
            auto token = ObjectLoaderLexer(&semantic, &location, m_Driver);
            switch(token)
            {
            case ObjFileToken::Integer:
            {
                start_tc_ind = semantic.IntegerValue;
                auto token = ObjectLoaderLexer(&semantic, &location, m_Driver);
                switch(token)
                {
                case ObjFileToken::Integer: // TODO: guarantee 3
                {
                    prev_pos_ind = semantic.IntegerValue;
                    success &= expect(static_cast<ObjFileToken>('/'), &token), parseIndex(&prev_tc_ind, &token) && parseIndex(&cur_pos_ind, &token);
                    if(success)
                    {
                        do
                        {
                            success &=
                                expect(static_cast<ObjFileToken>('/'), &token) &&
                                parseIndex(&cur_tc_ind, &token);
                            if(!success)
                                break;

                            m_Driver.pushPositionIndex(start_pos_ind);
                            m_Driver.pushTexCoordIndex(start_tc_ind);

                            m_Driver.pushPositionIndex(prev_pos_ind);
                            m_Driver.pushTexCoordIndex(prev_tc_ind);

                            m_Driver.pushPositionIndex(cur_pos_ind);
                            m_Driver.pushTexCoordIndex(cur_tc_ind);

                            prev_pos_ind = cur_pos_ind;
                            prev_tc_ind = cur_tc_ind;

                            token = ObjectLoaderLexer(&semantic, &location, m_Driver);
                            cur_pos_ind = semantic.IntegerValue;
                        } while(token == ObjFileToken::Integer);
                        success &= IsEndToken(token);
                    }
                } break;
                case static_cast<ObjFileToken>('/'):
                {
                    success &= parseIndex(&start_norm_ind, &token);
                    if(success)
                    {
                        success &= parseIndex(&prev_pos_ind, &token) &&
                                   expect(static_cast<ObjFileToken>('/'), &token) &&
                                   parseIndex(&prev_tc_ind, &token) &&
                                   expect(static_cast<ObjFileToken>('/'), &token) &&
                                   parseIndex(&prev_norm_ind, &token) &&
                                   parseIndex(&cur_pos_ind, &token);
                        if(success)
                        {
                            do
                            {
                                success &=
                                    expect(static_cast<ObjFileToken>('/'), &token) &&
                                    parseIndex(&cur_tc_ind, &token) &&
                                    expect(static_cast<ObjFileToken>('/'), &token) &&
                                    parseIndex(&cur_norm_ind, &token);

                                if(!success)
                                    break;

                                m_Driver.pushPositionIndex(start_pos_ind);
                                m_Driver.pushTexCoordIndex(start_tc_ind);
                                m_Driver.pushNormalIndex(start_norm_ind);

                                m_Driver.pushPositionIndex(prev_pos_ind);
                                m_Driver.pushTexCoordIndex(prev_tc_ind);
                                m_Driver.pushNormalIndex(prev_norm_ind);

                                m_Driver.pushPositionIndex(cur_pos_ind);
                                m_Driver.pushTexCoordIndex(cur_tc_ind);
                                m_Driver.pushNormalIndex(cur_norm_ind);

                                prev_pos_ind = cur_pos_ind;
                                prev_tc_ind = cur_tc_ind;
                                prev_norm_ind = cur_norm_ind;

                                token = ObjectLoaderLexer(&semantic, &location, m_Driver);
                                cur_pos_ind = semantic.IntegerValue;
                            } while(token == ObjFileToken::Integer);
                            success &= IsEndToken(token);
                        }
                    }
                } break;
                default:
                {
                    std::stringstream ss;
                    ss << "Unexpected \"" << TranslateToken(token) << ".";
                    m_Driver.error(location, ss.str());
                    success = false;
                } break;
                }                
            } break;
            case static_cast<ObjFileToken>('/'):
            {
                success &= parseIndex(&start_norm_ind, &token);
                if(success)
                {
                    success &= parseIndex(&prev_pos_ind, &token) &&
                               expect(static_cast<ObjFileToken>('/'), &token) &&
                               expect(static_cast<ObjFileToken>('/'), &token) &&
                               parseIndex(&prev_norm_ind, &token) &&
                               parseIndex(&cur_pos_ind, &token);
                    if(success)
                    {
                        do
                        {
                            success &=
                                expect(static_cast<ObjFileToken>('/'), &token) &&
                                expect(static_cast<ObjFileToken>('/'), &token) &&
                                parseIndex(&cur_tc_ind, &token);

                            if(!success)
                                break;

                            m_Driver.pushPositionIndex(start_pos_ind);
                            m_Driver.pushNormalIndex(start_norm_ind);

                            m_Driver.pushPositionIndex(prev_pos_ind);
                            m_Driver.pushNormalIndex(prev_norm_ind);

                            m_Driver.pushPositionIndex(cur_pos_ind);
                            m_Driver.pushNormalIndex(cur_norm_ind);

                            prev_pos_ind = cur_pos_ind;
                            prev_norm_ind = cur_norm_ind;
                            token = ObjectLoaderLexer(&semantic, &location, m_Driver);
                            cur_pos_ind = semantic.IntegerValue;
                        } while(token == ObjFileToken::Integer);
                        success &= IsEndToken(token);
                    }
                }
            } break;
            default:
            {
                std::stringstream ss;
                ss << "Unexpected \"" << TranslateToken(token) << ".";
                m_Driver.error(location, ss.str());
                success = false;
            } break;
            } 
        } break;
        default:
        {
            std::stringstream ss;
            ss << "Unexpected \"" << TranslateToken(token) << ".";
            m_Driver.error(location, ss.str());
            success = false;
        } break;
        }
    }

    if(!success)
    {
        m_Driver.error(declaration_location, "\tValid vertex declarations are:\n"
                       "\t\tf <index> <index> <index> [ <index> ... ]\n"
                       "\t\tf <index>/<index> <index>/<index> <index>/<index> [ <index>/<index> ... ] \n"
                       "\t\tf <index>/<index>/<index> <index>/<index>/<index> <index>/<index>/<index> [ <index>/<index>/<index> ... ] \n");
        return;
    }
}

int Parser::parse()
{
    ObjFileToken token;
    SemanticType semantic;
    Location location;
    while((token = ObjectLoaderLexer(&semantic, &location, m_Driver)) != ObjFileToken::EndOfFile)
    {
        switch(token)
        {
        case ObjFileToken::EndOfLine:
        {
            // --ignore
        } break;
        case ObjFileToken::VertGeom /* "v" */:
        {
            float x, y, z, w = 1.0f;
            auto success = parseNumber(&x, &token) && parseNumber(&y, &token) && parseNumber(&z, &token);

            if(success)
            {
                token = ObjectLoaderLexer(&semantic, &location, m_Driver);
                if(IsNumberToken(token))
                {
                    ConvertNumber(token, &semantic, &w);
                }
                else
                {
                    success &= ExpectEnd(m_Driver, location, token);
                }
            }

            if(!success)
            {
                skipToEndOfLine(token);
                m_Driver.error(location, "\tVertex position should be declared as follows: v <number> <number> <number> [ <number> ].");
                break;
            }

            m_Driver.pushPosition(x, y, z, w);

        } break;
        case ObjFileToken::VertNormal /* vn */:
        {
            float x, y, z;
            auto success = parseNumber(&x, &token) && parseNumber(&y, &token) && parseNumber(&z, &token);
            if(success)
            {
                token = ObjectLoaderLexer(&semantic, &location, m_Driver);
                ExpectEnd(m_Driver, location, token);
            }

            if(!success)
            {
                skipToEndOfLine(token);
                m_Driver.error(location, "\tVertex normal should be declared as follows: vn <number> <number> <number>.");
                break;
            }

            m_Driver.pushNormal(x, y, z);
        } break;
        case ObjFileToken::VertTexCoord /* vt */:
        {
            float x, y;
            auto success = parseNumber(&x, &token) && parseNumber(&y, &token);

            if(success)
            {
                token = ObjectLoaderLexer(&semantic, &location, m_Driver);
                if(IsNumberToken(token))
                {
                    token = ObjectLoaderLexer(&semantic, &location, m_Driver);
                }

                success &= ExpectEnd(m_Driver, location, token);
            }

            if(!success)
            {
                skipToEndOfLine(token);
                m_Driver.error(location, "\tVertex texture coordinate should be declared as follows: vn <number> <number> [ <number> ].");
                break;
            }

            m_Driver.pushTexCoord(x, y);
        } break;
        case ObjFileToken::GroupName /* g */:
        {
            const char* str;
            auto success = parseString(&semantic, &str, &token);

            if(!success)
            {
                skipToEndOfLine(token);
                m_Driver.error(location, "\tMesh group should be defined as follows: g <string>.");
                break;
            }

            m_Driver.pushGroup(str);
        } break;
        case ObjFileToken::Face /* f */:
        {
            parseIndices(location);
        } break;
        case ObjFileToken::MaterialName /* usemtl */:
        {
            const char* str;
            auto success = parseString(&semantic, &str, &token);

            if(!success)
            {
                skipToEndOfLine(token);
                m_Driver.error(location, "\tMaterial name should be defined as follows: usemtl <string>.");
                break;
            }

            m_Driver.pushMaterial(location, str);
        } break;
        case ObjFileToken::MaterialLibrary /* mtllib */:
        {
            const char* str;
            auto success = parseString(&semantic, &str, &token);

            if(!success)
            {
                m_Driver.error(location, "\tMaterial library should be defined as follows: usemtl <string>.");
                break;
            }

            m_Driver.parseMaterialFile(location, str);
        } break;
        default:
        {
            std::stringstream ss;
            ss << "Unexpected \"", TranslateToken(token), "\". It is not a valid top level token or it is currently unsupported.";
            m_Driver.error(location, ss.str());
        }
        }
    }
    return 0;
}
}
}