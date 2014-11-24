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

%{
#include "tempest/utils/memory.hh"
#include "tempest/utils//logging.hh"
#include "tempest/shader/shader-driver.hh"
#include "shader-parser.hh"

#undef yywrap
#define yywrap() 1

#define yyterminate() return Parser::token::T_END
%}

%option noyywrap nounput batch
%option never-interactive
%option prefix="shader_"

%{
#define YY_USER_ACTION  yylloc->columns(yyleng);

using namespace Tempest::Shader;
using namespace Tempest;
%}

INTEGER     \-{0,1}[0-9]+
MANTISA     (e|E)(\+|\-)?[0-9]*
IDENTIFIER  [_a-zA-Z][_a-zA-Z0-9]*

%x C_COMMENT
%x CPP_COMMENT

%%
%{
    yylloc->step();
%}

{INTEGER}"."[0-9]+{MANTISA}?              |
{INTEGER}{MANTISA}                      *yylval = CreateNode<Value<float>>(ToLocation(*yylloc), static_cast<float>(atof(yytext))); return Parser::token::T_FLOAT;
{INTEGER}u                              *yylval = CreateNode<Value<unsigned>>(ToLocation(*yylloc), strtoul(yytext, 0, 0)); return Parser::token::T_UNSIGNED;
{INTEGER}                               *yylval = CreateNode<Value<int>>(ToLocation(*yylloc), atoi(yytext)); return Parser::token::T_INTEGER;
"import"                                *yylval = AST::Node(); return Parser::token::T_IMPORT;
"if"                                    *yylval = AST::Node(); return Parser::token::T_IF;
"else"                                  *yylval = AST::Node(); return Parser::token::T_ELSE;
"break"                                 *yylval = AST::Node(); return Parser::token::T_BREAK;
"return"                                *yylval = AST::Node(); return Parser::token::T_RETURN;
"continue"                              *yylval = AST::Node(); return Parser::token::T_CONTINUE;
"do"                                    *yylval = AST::Node(); return Parser::token::T_DO;
"for"                                   *yylval = AST::Node(); return Parser::token::T_FOR;
"while"                                 *yylval = AST::Node(); return Parser::token::T_WHILE;
"switch"                                *yylval = AST::Node(); return Parser::token::T_SWITCH;
"case"                                  *yylval = AST::Node(); return Parser::token::T_CASE;
"default"                               *yylval = AST::Node(); return Parser::token::T_DEFAULT;
"true"                                  *yylval = CreateNode<Value<bool>>(ToLocation(*yylloc), true); return Parser::token::T_BOOLEAN;
"false"                                 *yylval = CreateNode<Value<bool>>(ToLocation(*yylloc), false); return Parser::token::T_BOOLEAN;
"void"                                  *yylval = AST::Node(); return Parser::token::T_TYPE_VOID;
"pass"                                  *yylval = AST::Node(); return Parser::token::T_PASS;
"technique"                             *yylval = AST::Node(); return Parser::token::T_TECHNIQUE;
"shader"                                *yylval = AST::Node(); return Parser::token::T_SHADER;
"vertex"                                *yylval = AST::Node(); return Parser::token::T_VERTEX_QUALIFIER;
"fragment"                              *yylval = AST::Node(); return Parser::token::T_FRAGMENT_QUALIFIER;
"layout"                                *yylval = AST::Node(); return Parser::token::T_LAYOUT_QUALIFIER;
"inout"                                 *yylval = AST::Node(); return Parser::token::T_INOUT_QUALIFIER;
"in"                                    *yylval = AST::Node(); return Parser::token::T_IN_QUALIFIER;
"out"                                   *yylval = AST::Node(); return Parser::token::T_OUT_QUALIFIER;
"centroid"                              *yylval = AST::Node(); return Parser::token::T_CENTROID_QUALIFIER;
"sample"                                *yylval = AST::Node(); return Parser::token::T_SAMPLE_QUALIFIER;
"invariant"                             *yylval = AST::Node(); return Parser::token::T_INVARIANT_QUALIFIER;
"buffer"                                *yylval = AST::Node(); return Parser::token::T_BUFFER_QUALIFIER;
"constant"                              *yylval = AST::Node(); return Parser::token::T_CONSTANT_QUALIFIER;
"const"                                 *yylval = AST::Node(); return Parser::token::T_CONST_QUALIFIER;
"uniform"                               *yylval = AST::Node(); return Parser::token::T_UNIFORM_QUALIFIER;
"struct"                                *yylval = AST::Node(); return Parser::token::T_STRUCT_QUALIFIER;
"highp"                                 *yylval = AST::Node(); return Parser::token::T_HIGHP_QUALIFIER;
"mediump"                               *yylval = AST::Node(); return Parser::token::T_MEDIUMP_QUALIFIER;
"lowp"                                  *yylval = AST::Node(); return Parser::token::T_LOWP_QUALIFIER;
"flat"                                  *yylval = AST::Node(); return Parser::token::T_FLAT_QUALIFIER;
"smooth"                                *yylval = AST::Node(); return Parser::token::T_SMOOTH_QUALIFIER;
"noperspective"                         *yylval = AST::Node(); return Parser::token::T_NOPERSPECTIVE_QUALIFIER;
"resource"                              *yylval = AST::Node(); return Parser::token::T_RESOURCE_QUALIFIER;
"structbuffer"                          *yylval = AST::Node(); return Parser::token::T_STRUCTBUFFER_QUALIFIER;
{IDENTIFIER}                            {
                                            auto node = driver.findIdentifier(yytext);
                                            if(node)
                                            {
                                                switch(node->getNodeType())
                                                {
                                                case TGE_EFFECT_FUNCTION_SET: *yylval = CreateNode<FunctionSetRef>(ToLocation(*yylloc), node->extract<FunctionSet>()); return Parser::token::T_FUNCTION;
                                                case TGE_EFFECT_VARIABLE: *yylval = CreateNode<VariableRef>(ToLocation(*yylloc), node->extract<Variable>()); return Parser::token::T_VARIABLE;
                                                case TGE_EFFECT_TYPE: *yylval = CreateNode<TypeRef>(ToLocation(*yylloc), node->extract<Type>()); return Parser::token::T_TYPE;
                                                case TGE_EFFECT_TYPEDEF: *yylval = CreateNode<TypeRef>(ToLocation(*yylloc), node->extract<Typedef>()->getType()); return Parser::token::T_TYPE;
                                                default:
                                                    TGE_ASSERT(false, "Unexpected type bound to identifier"); break;
                                                }
                                            }
                                            *yylval = CreateNode<Value<string>>(ToLocation(*yylloc), yytext);
                                            return Parser::token::T_IDENTIFIER;
                                        }
"+="                                    *yylval = AST::Node(); return Parser::token::T_ADD_ASSIGN;
"-="                                    *yylval = AST::Node(); return Parser::token::T_SUB_ASSIGN;
"*="                                    *yylval = AST::Node(); return Parser::token::T_MUL_ASSIGN;
"/="                                    *yylval = AST::Node(); return Parser::token::T_DIV_ASSIGN;
"%="                                    *yylval = AST::Node(); return Parser::token::T_BITWISE_MOD_ASSIGN;
"&="                                    *yylval = AST::Node(); return Parser::token::T_BITWISE_AND_ASSIGN;
"^="                                    *yylval = AST::Node(); return Parser::token::T_BITWISE_XOR_ASSIGN;
"|="                                    *yylval = AST::Node(); return Parser::token::T_BITWISE_OR_ASSIGN;
"||"                                    *yylval = AST::Node(); return Parser::token::T_OR;
"&&"                                    *yylval = AST::Node(); return Parser::token::T_AND;
"=="                                    *yylval = AST::Node(); return Parser::token::T_EQUAL;
"!="                                    *yylval = AST::Node(); return Parser::token::T_NEQUAL;
"<="                                    *yylval = AST::Node(); return Parser::token::T_LEQUAL;
">="                                    *yylval = AST::Node(); return Parser::token::T_GEQUAL;
"++"                                    *yylval = AST::Node(); return Parser::token::T_INCR;
"--"                                    *yylval = AST::Node(); return Parser::token::T_DECR;
"^^"                                    *yylval = AST::Node(); return Parser::token::T_XOR;
">>"                                    *yylval = AST::Node(); return Parser::token::T_SHIFT_RIGHT;
"<<"                                    *yylval = AST::Node(); return Parser::token::T_SHIFT_LEFT;
\"(\\.|[^\\"])*\"                       {
                                            yytext[strlen(yytext)-1] = 0;
                                            *yylval = CreateNode<StringLiteral>(ToLocation(*yylloc), yytext+1);
                                            return Parser::token::T_STRING_LITERAL;
                                        }
[,;:?=|\^&><\+\-\*/%!~\.\(\)\[\]\{\}]   *yylval = AST::Node(); return static_cast<Parser::token_type>(yytext[0]);
"//"                                    { BEGIN(CPP_COMMENT); }
<CPP_COMMENT>\n                         { BEGIN(INITIAL); yylloc->lines(yyleng); }
<CPP_COMMENT>.                          { }
"/*"                                    { BEGIN(C_COMMENT); }
<C_COMMENT>"*/"                         { BEGIN(INITIAL); }
<C_COMMENT>\n                           yylloc->lines(yyleng);
<C_COMMENT>.                            { }
[ \t]+                                  yylloc->step();
[\r\n]+                                 yylloc->lines(yyleng); yylloc->step();

.                                       driver.error(ToLocation(*yylloc), string("invalid character: ") + yytext);

%%

namespace Tempest
{
namespace Shader
{
#define SCAN_BYTES shader__scan_bytes
#define DELETE_BUFFER shader__delete_buffer
#define SWITCH_TO_BUFFER shader__switch_to_buffer
#include "shader/ast-driver-parse.impl.cc"
}
}
