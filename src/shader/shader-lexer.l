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
#include "tempest/shader/shader-parser.hh"

#undef yywrap
#define yywrap() 1

#define yyterminate() return ShaderToken::EndOfFile
%}

%option noyywrap nounput batch
%option never-interactive
%option prefix="shader_"

%{
#define YY_USER_ACTION  yylloc->startColumn += yyleng;

using namespace Tempest::Shader;
using namespace Tempest;
%}

INTEGER     \-{0,1}[0-9]+
MANTISA     (e|E)(\+|\-)?[0-9]*
IDENTIFIER  [_a-zA-Z][_a-zA-Z0-9]*

%x C_COMMENT
%x CPP_COMMENT

%%

{INTEGER}"."[0-9]+{MANTISA}?              |
{INTEGER}{MANTISA}                      *yylval = CreateNode<Value<float>>(*yylloc, static_cast<float>(atof(yytext))); return ShaderToken::Float;
{INTEGER}u                              *yylval = CreateNode<Value<unsigned>>(*yylloc, strtoul(yytext, 0, 0)); return ShaderToken::Unsigned;
{INTEGER}                               *yylval = CreateNode<Value<int>>(*yylloc, strtol(yytext, nullptr, 0)); return ShaderToken::Integer;
"0x"[0-9A-F]+                           *yylval = CreateNode<Value<int>>(*yylloc, strtol(yytext, nullptr, 16)); return ShaderToken::Integer;
"0x"[0-9A-F]+u                          *yylval = CreateNode<Value<int>>(*yylloc, strtoul(yytext, nullptr, 16)); return ShaderToken::Unsigned;
"import"                                *yylval = AST::Node(); return ShaderToken::Import;
"if"                                    *yylval = AST::Node(); return ShaderToken::If;
"else"                                  *yylval = AST::Node(); return ShaderToken::Else;
"break"                                 *yylval = AST::Node(); return ShaderToken::Break;
"return"                                *yylval = AST::Node(); return ShaderToken::Return;
"continue"                              *yylval = AST::Node(); return ShaderToken::Continue;
"do"                                    *yylval = AST::Node(); return ShaderToken::Do;
"for"                                   *yylval = AST::Node(); return ShaderToken::For;
"while"                                 *yylval = AST::Node(); return ShaderToken::While;
"switch"                                *yylval = AST::Node(); return ShaderToken::Switch;
"case"                                  *yylval = AST::Node(); return ShaderToken::Case;
"default"                               *yylval = AST::Node(); return ShaderToken::Default;
"true"                                  *yylval = CreateNode<Value<bool>>(*yylloc, true); return ShaderToken::Boolean;
"false"                                 *yylval = CreateNode<Value<bool>>(*yylloc, false); return ShaderToken::Boolean;
"void"                                  *yylval = AST::Node(); return ShaderToken::VoidType;
"shader"                                *yylval = AST::Node(); return ShaderToken::Shader;
"vertex"                                *yylval = AST::Node(); return ShaderToken::VertexQualifier;
"fragment"                              *yylval = AST::Node(); return ShaderToken::FragmentQualifier;
"layout"                                *yylval = AST::Node(); return ShaderToken::LayoutQualifier;
"inout"                                 *yylval = AST::Node(); return ShaderToken::InOutQualifier;
"in"                                    *yylval = AST::Node(); return ShaderToken::InQualifier;
"out"                                   *yylval = AST::Node(); return ShaderToken::OutQualifier;
"centroid"                              *yylval = AST::Node(); return ShaderToken::CentroidQualifier;
"sample"                                *yylval = AST::Node(); return ShaderToken::SampleQualifier;
"invariant"                             *yylval = AST::Node(); return ShaderToken::InvariantQualifier;
"buffer"                                *yylval = AST::Node(); return ShaderToken::BufferQualifier;
"constant"                              *yylval = AST::Node(); return ShaderToken::ConstantQualifier;
"const"                                 *yylval = AST::Node(); return ShaderToken::ConstQualifier;
"uniform"                               *yylval = AST::Node(); return ShaderToken::UniformQualifier;
"struct"                                *yylval = AST::Node(); return ShaderToken::StructQualifier;
"highp"                                 *yylval = AST::Node(); return ShaderToken::HighpQualifier;
"mediump"                               *yylval = AST::Node(); return ShaderToken::MediumpQualifier;
"lowp"                                  *yylval = AST::Node(); return ShaderToken::LowpQualifier;
"flat"                                  *yylval = AST::Node(); return ShaderToken::FlatQualifier;
"smooth"                                *yylval = AST::Node(); return ShaderToken::SmoothQualifier;
"noperspective"                         *yylval = AST::Node(); return ShaderToken::NoPerspectiveQualifier;
"resource"                              *yylval = AST::Node(); return ShaderToken::ResourceQualifier;
"structbuffer"                          *yylval = AST::Node(); return ShaderToken::StructBufferQualifier;
"options"                               *yylval = AST::Node(); return ShaderToken::Options;
{IDENTIFIER}                            {
                                            auto node = driver.findIdentifier(yytext);
                                            if(node)
                                            {
                                                switch(node->getNodeType())
                                                {
                                                case TGE_EFFECT_FUNCTION_SET: *yylval = CreateNode<FunctionSetRef>(*yylloc, node->extract<FunctionSet>()); return ShaderToken::Function;
                                                case TGE_EFFECT_VARIABLE: *yylval = CreateNode<VariableRef>(*yylloc, node->extract<Variable>()); return ShaderToken::Variable;
                                                case TGE_EFFECT_TYPE: *yylval = CreateNode<TypeRef>(*yylloc, node->extract<Type>()); return ShaderToken::Type;
                                                case TGE_EFFECT_TYPEDEF: *yylval = CreateNode<TypeRef>(*yylloc, node->extract<Typedef>()->getType()); return ShaderToken::Type;
                                                case TGE_EFFECT_OPTION: *yylval = CreateNode<OptionRef>(*yylloc, node->extract<Option>()); return ShaderToken::Option;
                                                default:
                                                    TGE_ASSERT(false, "Unexpected type bound to identifier"); break;
                                                }
                                            }
                                            *yylval = CreateNode<Value<std::string>>(*yylloc, yytext);
                                            return ShaderToken::Identifier;
                                        }
"+="                                    *yylval = AST::Node(); return ShaderToken::AddAssign;
"-="                                    *yylval = AST::Node(); return ShaderToken::SubAssign;
"*="                                    *yylval = AST::Node(); return ShaderToken::MulAssign;
"/="                                    *yylval = AST::Node(); return ShaderToken::DivAssign;
"%="                                    *yylval = AST::Node(); return ShaderToken::ModAssign;
"&="                                    *yylval = AST::Node(); return ShaderToken::BitwiseAndAssign;
"^="                                    *yylval = AST::Node(); return ShaderToken::BitwiseXorAssign;
"|="                                    *yylval = AST::Node(); return ShaderToken::BitwiseOrAssign;
"||"                                    *yylval = AST::Node(); return ShaderToken::LogicalOr;
"&&"                                    *yylval = AST::Node(); return ShaderToken::LogicalAnd;
"=="                                    *yylval = AST::Node(); return ShaderToken::Equal;
"!="                                    *yylval = AST::Node(); return ShaderToken::NotEqual;
"<="                                    *yylval = AST::Node(); return ShaderToken::LessEqual;
">="                                    *yylval = AST::Node(); return ShaderToken::GreaterEqual;
"++"                                    *yylval = AST::Node(); return ShaderToken::Increment;
"--"                                    *yylval = AST::Node(); return ShaderToken::Decrement;
"^^"                                    *yylval = AST::Node(); return ShaderToken::LogicalXor;
">>"                                    *yylval = AST::Node(); return ShaderToken::ShiftRight;
"<<"                                    *yylval = AST::Node(); return ShaderToken::ShiftLeft;
\"(\\.|[^\\"])*\"                       {
                                            yytext[strlen(yytext)-1] = 0;
                                            *yylval = CreateNode<StringLiteral>(*yylloc, yytext+1);
                                            return ShaderToken::StringLiteral;
                                        }
[,;:?=|\^&>\@<\+\-\*/%!~\.\(\)\[\]\{\}] *yylval = AST::Node(); return ToCharacterToken(yytext[0]);
"//"                                    { BEGIN(CPP_COMMENT); }
<CPP_COMMENT>\n                         { BEGIN(INITIAL); yylloc->startLine += yyleng; yylloc->startColumn = 1; }
<CPP_COMMENT>.                          { }
"/*"                                    { BEGIN(C_COMMENT); }
<C_COMMENT>"*/"                         { BEGIN(INITIAL); }
<C_COMMENT>\n                           yylloc->startLine += yyleng;
<C_COMMENT>.                            { }
[ \t]+                                  
[\r\n]+                                 yylloc->startLine += yyleng; yylloc->startColumn = 1;

.                                       driver.error(*yylloc, std::string("invalid character: ") + yytext);

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
