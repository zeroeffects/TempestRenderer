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

#include <cstdint>
#include "tempest/mesh/obj-mtl-loader-driver.hh"

namespace Tempest
{
namespace ObjMtlLoader
{
enum class ObjMtlToken: uint32_t
{
#define OBJ_MTL_TOKEN(token_enum, token_name) token_enum,
#include "tempest/mesh/obj-mtl-tokens.hh"
#undef OBJ_MTL_TOKEN
};
    
#define YY_DECL Tempest::ObjMtlLoader::ObjMtlToken ObjMtlLoaderLexer(Tempest::ObjMtlLoader::SemanticType* yylval,  \
                                                                     Tempest::Location* yylloc, \
                                                                     Tempest::ObjMtlLoader::Driver& driver)    

union SemanticType
{
    char StringValue[128];
    float FloatValue;
    int IntegerValue;
    struct
    {
        uint32_t PositionIndex[2],
                 TexCoordIndex[2],
                 NormalIndex[2];
    } TemporaryPair;
};

class Parser
{
    ObjMtlLoader::Driver& m_Driver;

    Location              m_CurrentLocation = Location{ nullptr, 1, 1 };
public:
    Parser(ObjMtlLoader::Driver& driver);
    
    int parse();
private:
    void skipToEndOfLine(ObjMtlToken cur_token);
    bool parseNumber(float* result, ObjMtlToken* res_token);
    bool parseString(SemanticType* semantic, const char** result, ObjMtlToken* res_token);
};
}
}