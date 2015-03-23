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
 
%skeleton "lalr1.cc"
%require "2.4"
%define parser_class_name "Parser"
%define namespace "Tempest::ObjLoader"

%union {
    char StringValue[128];
    float FloatValue;
    int IntegerValue;
    struct
    {
        uint32 PositionIndex[2],
               TexCoordIndex[2],
               NormalIndex[2];
    } TemporaryPair;
}

%code requires
{
#include "tempest/mesh/obj-loader-driver.hh"
}

%name-prefix="obj_loader_"
%parse-param { Driver& driver }
%lex-param { Driver& driver }

%locations
%initial-action
{
    @$.begin.filename = @$.end.filename = &driver.__FileName;
};

//%debug
%error-verbose

%token          T_EOF                0   "end of file";
%token          T_EOL                    "end of line";

%token <StringValue>    T_STRING         "string";
%token <IntegerValue>   T_INTEGER        "integer";
%token <FloatValue>     T_FLOAT          "float";
%token          T_VERT_PARAM             "vp";
%token          T_VERT_NORMAL            "vn";
%token          T_VERT_TEXCOORD          "vt";
%token          T_VERT_GEOM              "v";
%token          T_DEGREE                 "deg";
%token          T_BASIS_MATRIX           "bmat";
%token          T_STEP                   "step";
%token          T_CURVE_SURFACE_TYPE     "cstype";
%token          T_PARAM_VALUE            "parm";
%token          T_TRIM                   "trim";
%token          T_HOLE                   "hole";
%token          T_SPECIAL_CURVE          "scrv";
%token          T_SPECIAL_POINT          "sp";
%token          T_END                    "end";
%token          T_CONNECT                "con";
%token          T_POINT                  "p";
%token          T_LINE                   "l";
%token          T_FACE                   "f";
%token          T_CURVE                  "curv";
%token          T_CURVE_2D               "curv2";
%token          T_SURFACE                "surf";
%token          T_GROUP_NAME             "g";
%token          T_SMOOTHING_GROUP        "s";
%token          T_MERGING_GROUP          "mg";
%token          T_OBJECT_NAME            "o";
%token          T_BEVEL_INTERP           "bevel";
%token          T_COLOR_INTERP           "c_interp";
%token          T_DISSOLVE_INTERP        "d_interp";
%token          T_LOD                    "lod";
%token          T_MATERIAL_NAME          "usemtl";
%token          T_MATERIAL_LIBRARY       "mtllib";
%token          T_SHADOW_CASTING         "shadow_obj";
%token          T_RAY_TRACING            "trace_obj";
%token          T_CURVE_APPROX           "ctech";
%token          T_SURFACE_APPROX         "stech";
%token          T_OFF                    "off";

%type <TemporaryPair> face_P face_PN face_PT face_PTN
%type <FloatValue> number

%start obj_loader_file

%code provides
{
#define YY_DECL Tempest::ObjLoader::Parser::token_type obj_loader_lex(Tempest::ObjLoader::Parser::semantic_type* yylval,  \
                                                                      Tempest::ObjLoader::Parser::location_type* yylloc, \
                                                                      Tempest::ObjLoader::Driver& driver)
                                                                      
YY_DECL;

namespace Tempest
{
namespace ObjLoader
{
inline Location ToLocation(const location& loc) { return Location{ loc.begin.filename, loc.begin.line, loc.begin.column }; }
}
}
}

%%

obj_loader_file
    : /* empty */
    | command T_EOL obj_loader_file
    ;

command
    : /* empty */
    | "v" number number number                  { driver.pushPosition($2, $3, $4); }
    | "v" number number number number           { driver.pushPosition($2, $3, $4, $5); }
    | "vn" number number number                 { driver.pushNormal($2, $3, $4); }
    | "vt" number number                        { driver.pushTexCoord($2, $3); }
    | "vt" number number number                 { driver.pushTexCoord($2, $3); } // Discard z because 3D textures are useless.
    | "g" "string"                              { driver.pushGroup($2); }
    | "f" face_indices
    | "usemtl" "string"                         { driver.pushMaterial(ToLocation(@$), $2); }
    | "mtllib" "string"                         { driver.parseMaterialFile(ToLocation(@$), $2); }
    | "s" "integer"                             // IGNORED                  
    | "mg" "integer"                            // IGNORED
    | "s" "off"                                 // IGNORED
    // | "vp" UNSUPPORTED
    // | "deg" UNSUPPORTED
    // | "bmat" UNSUPPORTED
    // | "step" UNSUPPORTED
    // | "cstype" UNSUPPORTED
    // | "parm" UNSUPPORTED
    // | "trim" UNSUPPORTED
    // | "hole" UNSUPPORTED
    // | "scrv" UNSUPPORTED
    // | "sp" UNSUPPORTED
    // | "end" UNSUPPORTED
    // | "con" UNSUPPORTED
    // | "p" UNSUPPORTED
    // | "l" UNSUPPORTED
    // | "curv" UNSUPPORTED
    // | "curv2" UNSUPPORTED
    // | "surf" UNSUPPORTED
    // | "o" UNSUPPORTED
    // | "bevel" UNSUPPORTED
    // | "c_interp" UNSUPPORTED
    // | "d_interp" UNSUPPORTED
    // | "lod" UNSUPPORTED
    // | "shadow_obj" UNSUPPORTED
    // | "trace_obj" UNSUPPORTED
    // | "ctech" UNSUPPORTED
    // | "stech" UNSUPPORTED
    ;

number
    : "float"                                   { $$ = $1; }
    | "integer"                                 { $$ = static_cast<float>($1); }
    ;

face_indices
    : face_P
    | face_PT
    | face_PTN
    | face_PN
    ;
    
face_P
    : "integer" "integer" "integer"
                                            {
                                                driver.pushPositionIndex($1);
                                                driver.pushPositionIndex($2);
                                                driver.pushPositionIndex($3);
                                                $$.PositionIndex[0] = $2; 
                                                $$.PositionIndex[1] = $3;
                                            }
    | face_P "integer"                      {
                                                driver.pushPositionIndex($1.PositionIndex[0]);
                                                driver.pushPositionIndex($1.PositionIndex[1]);
                                                driver.pushPositionIndex($2);
                                                $$.PositionIndex[0] = $1.PositionIndex[1];
                                                $$.PositionIndex[1] = $2;
                                            }
    ;
    
face_PT
    : "integer" '/' "integer" "integer" '/' "integer" "integer" '/' "integer"
                                            {
                                                driver.pushPositionIndex($1); driver.pushTexCoordIndex($3);
                                                driver.pushPositionIndex($4); driver.pushTexCoordIndex($6);
                                                driver.pushPositionIndex($7); driver.pushTexCoordIndex($9);
                                                $$.PositionIndex[0] = $4; 
                                                $$.TexCoordIndex[0] = $6;
                                                $$.PositionIndex[1] = $7;
                                                $$.TexCoordIndex[1] = $9;
                                            }
    | face_PT "integer" '/' "integer"
                                            {
                                                driver.pushPositionIndex($1.PositionIndex[0]); driver.pushTexCoordIndex($1.TexCoordIndex[0]);
                                                driver.pushPositionIndex($1.PositionIndex[1]); driver.pushTexCoordIndex($1.TexCoordIndex[1]);
                                                driver.pushPositionIndex($2); driver.pushTexCoordIndex($4);
                                                $$.PositionIndex[0] = $1.PositionIndex[1];
                                                $$.TexCoordIndex[0] = $1.TexCoordIndex[1];
                                                $$.PositionIndex[1] = $2;
                                                $$.TexCoordIndex[1] = $4;
                                            }
    ;
face_PTN
    : "integer" '/' "integer" '/' "integer" "integer" '/' "integer" '/' "integer" "integer" '/' "integer" '/' "integer"
                                            {
                                                driver.pushPositionIndex($1); driver.pushTexCoordIndex($3); driver.pushNormalIndex($5);
                                                driver.pushPositionIndex($6); driver.pushTexCoordIndex($8); driver.pushNormalIndex($10);
                                                driver.pushPositionIndex($11); driver.pushTexCoordIndex($13); driver.pushNormalIndex($15);
                                                $$.PositionIndex[0] = $6;
                                                $$.TexCoordIndex[0] = $8;
                                                $$.NormalIndex[0] = $10;
                                                $$.PositionIndex[1] = $11;
                                                $$.TexCoordIndex[1] = $13;
                                                $$.NormalIndex[1] = $15;
                                            }
    | face_PTN "integer" '/' "integer" '/' "integer"
                                            {
                                                driver.pushPositionIndex($1.PositionIndex[0]); driver.pushTexCoordIndex($1.TexCoordIndex[0]); driver.pushNormalIndex($1.NormalIndex[0]);
                                                driver.pushPositionIndex($1.PositionIndex[1]); driver.pushTexCoordIndex($1.TexCoordIndex[1]); driver.pushNormalIndex($1.NormalIndex[1]);
                                                driver.pushPositionIndex($2); driver.pushTexCoordIndex($4); driver.pushNormalIndex($6);
                                                $$.PositionIndex[0] = $1.PositionIndex[1];
                                                $$.TexCoordIndex[0] = $1.TexCoordIndex[1];
                                                $$.NormalIndex[0] = $1.NormalIndex[1];
                                                $$.PositionIndex[1] = $2;
                                                $$.TexCoordIndex[1] = $4;
                                                $$.NormalIndex[1] = $6;
                                            }
    ;
 
face_PN
    : "integer" '/' '/' "integer" "integer" '/' '/' "integer" "integer" '/' '/' "integer"
                                            {
                                                driver.pushPositionIndex($1); driver.pushNormalIndex($4);
                                                driver.pushPositionIndex($5); driver.pushNormalIndex($8);
                                                driver.pushPositionIndex($9); driver.pushNormalIndex($12);
                                                $$.PositionIndex[0] = $5;
                                                $$.NormalIndex[0] = $8;
                                                $$.PositionIndex[1] = $9;
                                                $$.NormalIndex[1] = $12;
                                            }
    | face_PN "integer" '/' '/' "integer"
                                            {
                                                driver.pushPositionIndex($1.PositionIndex[0]); driver.pushNormalIndex($1.NormalIndex[0]);
                                                driver.pushPositionIndex($1.PositionIndex[1]); driver.pushNormalIndex($1.NormalIndex[1]);
                                                driver.pushPositionIndex($2); driver.pushNormalIndex($5);
                                                $$.PositionIndex[0] = $1.PositionIndex[1];
                                                $$.NormalIndex[0] = $1.NormalIndex[1];
                                                $$.PositionIndex[1] = $2;
                                                $$.NormalIndex[1] = $5;
                                            }
    ;

    
    
%%

namespace Tempest
{
namespace ObjLoader
{
void Parser::error(const Parser::location_type& l, const std::string& m)
{
    driver.error(ToLocation(l), m);
}
}
}
