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
#include "tempest/mesh/obj-loader-parser.hh"

#undef yywrap
#define yywrap() 1

#define yyterminate() return ObjFileToken::EndOfFile
%}

%option noyywrap nounput batch
%option never-interactive
%option prefix="obj_loader_"

%{
#define YY_USER_ACTION  yylloc->startColumn += yyleng;

using namespace Tempest::ObjLoader;
using namespace Tempest;
%}

INTEGER     \-{0,1}[0-9]+
MANTISA     (e|E)(\+|\-)?[0-9]*
IDENTIFIER  [_a-zA-Z][_a-zA-Z0-9]*
COMMENT     \#.*[\r]*\n

%%

{INTEGER}"."[0-9]+{MANTISA}?              |
{INTEGER}{MANTISA}                      yylval->FloatValue = (float)atof(yytext); return ObjFileToken::Float;
{INTEGER}                               yylval->IntegerValue = atoi(yytext); return ObjFileToken::Integer;
"vp"                                    return ObjFileToken::VertParam;
"vn"                                    return ObjFileToken::VertNormal;
"vt"                                    return ObjFileToken::VertTexCoord;
"v"                                     return ObjFileToken::VertGeom;
"deg"                                   return ObjFileToken::Degree;
"bmat"                                  return ObjFileToken::BasisMatrix;
"step"                                  return ObjFileToken::Step;
"cstype"                                return ObjFileToken::CurveSurfaceType;
"parm"                                  return ObjFileToken::ParamValue;
"trim"                                  return ObjFileToken::Trim;
"hole"                                  return ObjFileToken::Hole;
"scrv"                                  return ObjFileToken::SpecialCurve;
"sp"                                    return ObjFileToken::SpecialPoint;
"end"                                   return ObjFileToken::End;
"con"                                   return ObjFileToken::Connect;
"p"                                     return ObjFileToken::Point;
"l"                                     return ObjFileToken::Line;
"f"                                     return ObjFileToken::Face;
"curv"                                  return ObjFileToken::Curve;
"curv2"                                 return ObjFileToken::Curve2D;
"surf"                                  return ObjFileToken::Surface;
"g"                                     return ObjFileToken::GroupName;
"s"                                     return ObjFileToken::SmoothingGroup;
"mg"                                    return ObjFileToken::MergingGroup;
"o"                                     return ObjFileToken::ObjectName;
"bevel"                                 return ObjFileToken::BevelInterp;
"c_interp"                              return ObjFileToken::ColorInterp;
"d_interp"                              return ObjFileToken::DissolveInterp;
"lod"                                   return ObjFileToken::LOD;
"usemtl"                                return ObjFileToken::MaterialName;
"mtllib"                                return ObjFileToken::MaterialLibrary;
"shadow_obj"                            return ObjFileToken::ShadowCasting;
"trace_obj"                             return ObjFileToken::RayTracing;
"ctech"                                 return ObjFileToken::CurveApprox;
"stech"                                 return ObjFileToken::SurfaceApprox;
"off"                                   return ObjFileToken::Off;
[/]                                     return static_cast<ObjFileToken>(yytext[0]);
[A-Z0-9a-z_.]+                          strncpy(yylval->StringValue, yytext, sizeof(yylval->StringValue) - 1); return ObjFileToken::String;
{COMMENT}                               yylloc->startLine += 1; yylloc->startColumn = 1;
[ \t]+                                  
[\r\n]+                                 yylloc->startLine += yyleng; yylloc->startColumn = 1; return ObjFileToken::EndOfLine;
.                                       driver.error(*yylloc, string("invalid character: ") + yytext);

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
