/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2014 Zdravko Velinov
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
#include "tempest/mesh/obj-mtl-loader-driver.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/utils/logging.hh"
#include "obj-mtl-loader-parser.hh"

#undef yywrap
#define yywrap() 1

#define yyterminate() return Parser::token::T_EOF
%}

%option noyywrap nounput batch
%option never-interactive
%option prefix="obj_mtl_loader_"

%{
#define YY_USER_ACTION  yylloc->columns(yyleng);

using namespace Tempest::ObjMtlLoader;
using namespace Tempest;
%}

INTEGER     \-{0,1}[0-9]+
MANTISA     (e|E)(\+|\-)?[0-9]*
IDENTIFIER  [_a-zA-Z][_a-zA-Z0-9]*
COMMENT     \#.*[\r]*\n

%%
%{
    yylloc->step();
%}

{INTEGER}"."[0-9]+{MANTISA}?              |
{INTEGER}{MANTISA}                      yylval->FloatValue = static_cast<float>(atof(yytext)); return Parser::token::T_FLOAT;
{INTEGER}                               yylval->IntegerValue = atoi(yytext); return Parser::token::T_INTEGER;
"newmtl"                                return Parser::token::T_NEWMTL;
"Ka"                                    return Parser::token::T_KA;
"Kd"                                    return Parser::token::T_KD;
"Ks"                                    return Parser::token::T_KS;
"Tf"                                    return Parser::token::T_TF;
"illum"                                 return Parser::token::T_ILLUM;
"Ns"                                    return Parser::token::T_NS;
"sharpness"                             return Parser::token::T_SHARPNESS;
"Ni"                                    return Parser::token::T_NI;
"map_Ka"                                return Parser::token::T_MAP_KA;
"map_Kd"                                return Parser::token::T_MAP_KD;
"map_Ks"                                return Parser::token::T_MAP_KS;
"map_Ns"                                return Parser::token::T_MAP_NS;
"map_d"                                 return Parser::token::T_MAP_D;
"d"                                     return Parser::token::T_D;
"Tr"                                    return Parser::token::T_D;
"Ke"                                    return Parser::token::T_KE;
illum_[0-9]                             yylval->IntegerValue = yytext[6]; return Parser::token::T_ILLUM_MODEL;
[A-Z0-9a-z_.]+                          strncpy(yylval->StringValue, yytext, sizeof(yylval->StringValue) - 1); return Parser::token::T_STRING;
{COMMENT}                               yylloc->lines(1); yylloc->step();
[ \t]+                                  yylloc->step();
[\r\n]+                                 yylloc->lines(yyleng); yylloc->step(); return Parser::token::T_EOL;
.                                       driver.error(ToLocation(*yylloc), string("invalid character: ") + yytext);

%%

namespace Tempest
{
namespace ObjMtlLoader
{
#define SCAN_BYTES obj_mtl_loader__scan_bytes
#define DELETE_BUFFER obj_mtl_loader__delete_buffer
#define SWITCH_TO_BUFFER obj_mtl_loader__switch_to_buffer
#include "shader/ast-driver-parse.impl.cc"
}
}