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
#include "tempest/mesh/obj-mtl-parser.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/utils/logging.hh"

#undef yywrap
#define yywrap() 1

#define yyterminate() return ObjMtlToken::EndOfFile
%}

%option noyywrap nounput batch
%option never-interactive
%option prefix="obj_mtl_loader_"

%{
#define YY_USER_ACTION  yylloc->startColumn += yyleng;

using namespace Tempest::ObjMtlLoader;
using namespace Tempest;
%}

INTEGER     \-{0,1}[0-9]+
MANTISA     (e|E)(\+|\-)?[0-9]*
IDENTIFIER  [_a-zA-Z][_a-zA-Z0-9]*
COMMENT     \#[^\r\n]*

%x NAME_SUB

%%

{INTEGER}"."[0-9]+{MANTISA}?              |
{INTEGER}{MANTISA}                      yylval->FloatValue = static_cast<float>(atof(yytext)); return ObjMtlToken::Float;
{INTEGER}                               yylval->IntegerValue = atoi(yytext); return ObjMtlToken::Integer;
"newmtl"                                return ObjMtlToken::NewMtl;
"Ka"                                    return ObjMtlToken::AmbientReflectivity;
"Kd"                                    return ObjMtlToken::DiffuseReflectivity;
"Ks"                                    return ObjMtlToken::SpecularReflectivity;
"Tf"                                    return ObjMtlToken::TransmissionFilter;
"illum"                                 return ObjMtlToken::Illum;
"Ns"                                    return ObjMtlToken::SpecularExponent;
"sharpness"                             return ObjMtlToken::ReflectionSharpness;
"Ni"                                    return ObjMtlToken::RefractionIndex;
map_Ka[ \t]                             BEGIN(NAME_SUB); return ObjMtlToken::AmbientReflectivityMap;
map_Kd[ \t]                             BEGIN(NAME_SUB); return ObjMtlToken::DiffuseReflectivityMap;
map_Ks[ \t]                             BEGIN(NAME_SUB); return ObjMtlToken::SpecularReflectivityMap;
map_Ns[ \t]                             BEGIN(NAME_SUB); return ObjMtlToken::SpecularExponentMap;
map_d[ \t]                              BEGIN(NAME_SUB); return ObjMtlToken::DissolveMap;
<NAME_SUB>\n                              |
<NAME_SUB>[\r]*\n                       BEGIN(INITIAL); yylloc->startLine += 1; yylloc->startColumn = 1; return ObjMtlToken::EndOfLine;
<NAME_SUB>[^\n]*                        strncpy(yylval->StringValue, yytext, sizeof(yylval->StringValue) - 1); return ObjMtlToken::String;
"d"                                     return ObjMtlToken::Dissolve;
"Tr"                                    return ObjMtlToken::Dissolve;
"Ke"                                    return ObjMtlToken::Emission;
illum_[0-9]                             yylval->IntegerValue = yytext[6]; return ObjMtlToken::IllumModel;
[A-Z0-9a-z_.]+                          strncpy(yylval->StringValue, yytext, sizeof(yylval->StringValue) - 1); return ObjMtlToken::String;
{COMMENT}                               
[ \t]+                                  
\n                                        |                                  
[\r]*\n                                 yylloc->startLine += 1; yylloc->startColumn = 1; return ObjMtlToken::EndOfLine;
.                                       driver.error(*yylloc, std::string("invalid character: ") + yytext);

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