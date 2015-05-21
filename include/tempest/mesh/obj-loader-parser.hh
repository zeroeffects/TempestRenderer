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

#include "tempest/utils/types.hh"

namespace Tempest
{
namespace ObjLoader
{
enum class ObjFileToken: uint32
{
#define OBJ_LOADER_TOKEN(token_enum, token_name) token_enum,
#include "tempest/mesh/obj-loader-tokens.hh"
#undef OBJ_LOADER_TOKEN
};

union SemanticType
{
    char StringValue[128];
    float FloatValue;
    int IntegerValue;
    struct
    {
        uint32 PositionIndex[2],
            TexCoordIndex[2],
            NormalIndex[2];
    } TemporaryPair;
};

#define YY_DECL Tempest::ObjLoader::ObjFileToken ObjectLoaderLexer(Tempest::ObjLoader::SemanticType* yylval,  \
                                                                   Tempest::Location* yylloc, \
                                                                   Tempest::ObjLoader::Driver& driver)

class Parser
{
    ObjLoader::Driver& m_Driver;
public:
    Parser(ObjLoader::Driver& driver);

    int parse();

private:
    void skipToEndOfLine(ObjFileToken cur_token);
    bool expect(ObjFileToken expected, ObjFileToken* res_token);
    bool parseIndex(int* result, ObjFileToken* res_token);
    bool parseNumber(float* result, ObjFileToken* res_token);
    void parseIndices(const Location& location);
    bool parseString(SemanticType* semantic, const char** result, ObjFileToken* res_token);
};
}
}