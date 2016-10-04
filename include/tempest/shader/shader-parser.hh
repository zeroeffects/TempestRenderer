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

#ifndef _SHADER_PARSER_HH_
#define _SHADER_PARSER_HH_

#include <cstdint>
#include "tempest/parser/ast.hh"
#include "tempest/shader/shader-ast.hh"
#include "tempest/parser/driver-base.hh"

#include <vector>

namespace Tempest
{
namespace Shader
{
enum class ShaderToken: uint32_t
{
#define SHADER_TOKEN(token_enum, token_name) token_enum,
#include "tempest/shader/shader-tokens.hh"
#undef SHADER_TOKEN
    Count
};

#define ToCharacterToken(_c) static_cast<ShaderToken>((uint32_t)ShaderToken::Count + (uint32_t)(_c) - '!')

#define YY_DECL Tempest::Shader::ShaderToken ShaderLexer(Tempest::AST::Node* yylval,  \
                                                         Tempest::Location* yylloc, \
                                                         Tempest::Shader::Driver& driver)

class Driver;

class Parser
{
    Shader::Driver&        m_Driver;
    ShaderToken            m_CurrentToken;
    AST::Node              m_CurrentNode;
    Location               m_CurrentLocation = Location{ nullptr, 1, 1 };
    bool                   m_Reprocess = false;

    std::vector<AST::Node> m_NodeStack;
public:
    Parser(Shader::Driver& driver);
    ~Parser() = default;

    int parse();
private:
    void parseToken();

    void skipDeclarationOrDefinition();
    void skipDefinition();
    void skipToToken(char char_token);
    bool expect(ShaderToken expected);
    AST::NodeT<List> functionArgList();
    AST::NodeT<Expression> expression();
    AST::NodeT<Expression> suffixExpression();
    AST::NodeT<Expression> prefixExpression();
    AST::NodeT<Expression> binaryExpression(AST::NodeT<Expression> (Parser::*func)());
    AST::NodeT<Expression> assignmentExpression();
    AST::NodeT<Expression> paranthesesExpression();
    AST::NodeT<Expression> conditionalExpression();
    AST::Node expectNode(ShaderToken expected);
    AST::Node expectRedefCheck(ShaderToken expected, ShaderToken redef);
    AST::NodeT<List> collapseStackToList(ListType _type, size_t top_element);
    bool removableExtDeclaration();
    bool options();
    bool import();
    bool shader();
    bool function();
    
    bool statement();
    AST::Node statementList();
    AST::Node layoutHeader();
    AST::NodeT<VariableRef> variable();
    bool basicVariableDeclaration();
    bool expressionStatement();
    bool globalVariable();
    bool shaderExtDeclaration();
    bool selectionStatement();
    bool iterationStatement();
    bool blockStatement();
    bool jumpStatement();
    bool structMembers();
    AST::NodeT<List> structBody();

    bool buffer();
    bool bufferDeclaration();
    bool structBufferDeclaration();

    bool structDeclaration();

    AST::NodeT<Expression> option();
    bool optional(bool (Parser::*func)());
};
}
}

#endif // _SHADER_PARSER_HH_