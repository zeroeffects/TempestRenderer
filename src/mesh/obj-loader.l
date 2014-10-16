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

#include "tempest/mesh/obj-loader-driver.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/utils/logging.hh"
#include "obj-loader-parser.hh"

#undef yywrap
#define yywrap() 1

#define yyterminate() return Parser::token::T_EOF
%}

%option noyywrap nounput batch
%option never-interactive
%option prefix="obj_loader_"

%{
#define YY_USER_ACTION  yylloc->columns(yyleng);

using namespace Tempest::ObjLoader;
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
{INTEGER}{MANTISA}                      yylval->FloatValue = atof(yytext); return Parser::token::T_FLOAT;
{INTEGER}                               yylval->IntegerValue = atoi(yytext); return Parser::token::T_INTEGER;
"vp"                                    return Parser::token::T_VERT_PARAM;
"vn"                                    return Parser::token::T_VERT_NORMAL;
"vt"                                    return Parser::token::T_VERT_TEXCOORD;
"v"                                     return Parser::token::T_VERT_GEOM;
"deg"                                   return Parser::token::T_DEGREE;
"bmat"                                  return Parser::token::T_BASIS_MATRIX;
"step"                                  return Parser::token::T_STEP;
"cstype"                                return Parser::token::T_CURVE_SURFACE_TYPE;
"parm"                                  return Parser::token::T_PARAM_VALUE;
"trim"                                  return Parser::token::T_TRIM;
"hole"                                  return Parser::token::T_HOLE;
"scrv"                                  return Parser::token::T_SPECIAL_CURVE;
"sp"                                    return Parser::token::T_SPECIAL_POINT;
"end"                                   return Parser::token::T_END;
"con"                                   return Parser::token::T_CONNECT;
"p"                                     return Parser::token::T_POINT;
"l"                                     return Parser::token::T_LINE;
"f"                                     return Parser::token::T_FACE;
"curv"                                  return Parser::token::T_CURVE;
"curv2"                                 return Parser::token::T_CURVE_2D;
"surf"                                  return Parser::token::T_SURFACE;
"g"                                     return Parser::token::T_GROUP_NAME;
"s"                                     return Parser::token::T_SMOOTHING_GROUP;
"mg"                                    return Parser::token::T_MERGING_GROUP;
"o"                                     return Parser::token::T_OBJECT_NAME;
"bevel"                                 return Parser::token::T_BEVEL_INTERP;
"c_interp"                              return Parser::token::T_COLOR_INTERP;
"d_interp"                              return Parser::token::T_DISSOLVE_INTERP;
"lod"                                   return Parser::token::T_LOD;
"usemtl"                                return Parser::token::T_MATERIAL_NAME;
"mtllib"                                return Parser::token::T_MATERIAL_LIBRARY;
"shadow_obj"                            return Parser::token::T_SHADOW_CASTING;
"trace_obj"                             return Parser::token::T_RAY_TRACING;
"ctech"                                 return Parser::token::T_CURVE_APPROX;
"stech"                                 return Parser::token::T_SURFACE_APPROX;
"off"                                   return Parser::token::T_OFF;
[/]                                     return static_cast<Parser::token_type>(yytext[0]);
[A-Z0-9a-z_.]+                          strncpy(yylval->StringValue, yytext, sizeof(yylval->StringValue) - 1); return Parser::token::T_STRING;
{COMMENT}                               yylloc->lines(1); yylloc->step();
[ \t]+                                  yylloc->step();
[\r\n]+                                 yylloc->lines(yyleng); yylloc->step(); return Parser::token::T_EOL;
.                                       driver.error(ToLocation(*yylloc), string("invalid character: ") + yytext);

%%

namespace Tempest
{
namespace ObjLoader
{
#define SCAN_BYTES obj_loader__scan_bytes
#define DELETE_BUFFER obj_loader__delete_buffer
#define SWITCH_TO_BUFFER obj_loader__switch_to_buffer
#include "shader/ast-driver-parse.impl.cc"
}
}
