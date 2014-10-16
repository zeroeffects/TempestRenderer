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
%define namespace "Tempest::ObjMtlLoader"

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
#include "tempest/mesh/obj-mtl-loader-driver.hh"
}

%name-prefix="obj_mtl_loader_"
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
%token                  T_NEWMTL         "newmtl";
%token                  T_KA             "Ka";
%token                  T_KD             "Kd";
%token                  T_KS             "Ks";
%token                  T_KE             "Ke";
%token                  T_TF             "Tf";
%token                  T_ILLUM          "illum";
%token <IntegerValue>   T_ILLUM_MODEL    "illum_model";
%token                  T_D              "d";
%token                  T_NS             "Ns";
%token                  T_SHARPNESS      "sharpness";
%token                  T_NI             "Ni";
%token                  T_MAP_KA         "map_Ka";
%token                  T_MAP_KD         "map_Kd";
%token                  T_MAP_KS         "map_Ks";
%token                  T_MAP_NS         "map_Ns";
%token                  T_MAP_D          "map_d";

%type <FloatValue> number

%start obj_mtl_loader_file

%code provides
{
#define YY_DECL Tempest::ObjMtlLoader::Parser::token_type obj_mtl_loader_lex(Tempest::ObjMtlLoader::Parser::semantic_type* yylval,  \
                                                                             Tempest::ObjMtlLoader::Parser::location_type* yylloc, \
                                                                             Tempest::ObjMtlLoader::Driver& driver)
                                                                      
YY_DECL;

#define SET_MATERIAL_FIELD(log_line, name, value) \
    if(driver.getCurrentMaterial()) { \
        driver.getCurrentMaterial()->name; \
    } else { \
        driver.error(ToLocation(log_line), "Failed to initialize material field \"" #name "\" because no material was specified."); \
    }

namespace Tempest
{
namespace ObjMtlLoader
{
inline Location ToLocation(const location& loc) { return Location{ loc.begin.filename, loc.begin.line, loc.begin.column }; }
}
}
}

%%

obj_mtl_loader_file
    : /* empty */
    | command T_EOL obj_mtl_loader_file
    ;

command
    : /* empty */
    | "newmtl" "string"             { driver.pushNewMaterial($2); }
    | "Ka" number number number     { SET_MATERIAL_FIELD(@$, AmbientReflectivity, Vector3($2, $3, $4)); }
    | "Kd" number number number     { SET_MATERIAL_FIELD(@$, DiffuseReflectivity, Vector3($2, $3, $4)); }
    | "Ks" number number number     { SET_MATERIAL_FIELD(@$, SpecularReflectivity, Vector3($2, $3, $4)); }
    | "Tf" number number number     { SET_MATERIAL_FIELD(@$, TransmissionFilter, Vector3($2, $3, $4)); }
    | "illum" "illum_model"         { SET_MATERIAL_FIELD(@$, IllumModel, $2); }
    | "illum" "integer"             { SET_MATERIAL_FIELD(@$, IllumModel, $2); }
    | "Ns" number                   { SET_MATERIAL_FIELD(@$, SpecularExponent, $2); }
    | "sharpness" number            { SET_MATERIAL_FIELD(@$, ReflectionSharpness, $2); }
    | "Ni" number                   { SET_MATERIAL_FIELD(@$, RefractionIndex, $2); }
    | "d" number                    { SET_MATERIAL_FIELD(@$, Dissolve, $2); }
    | "d" number number             { SET_MATERIAL_FIELD(@$, Dissolve, $2); } // ignore $3
    | "Ke" number number number     { SET_MATERIAL_FIELD(@$, Emission, $2); }
    | "map_Ka" "string"             { SET_MATERIAL_FIELD(@$, AmbientReflectivityMap, $2); }
    | "map_Kd" "string"             { SET_MATERIAL_FIELD(@$, DiffuseReflectivityMap, $2); }
    | "map_Ks" "string"             { SET_MATERIAL_FIELD(@$, SpecularReflectivityMap, $2); }
    | "map_Ns" "string"             { SET_MATERIAL_FIELD(@$, SpecularExponentMap, $2); }
    | "map_d" "string"              { SET_MATERIAL_FIELD(@$, DissolveMap, $2); }
    ;

number
    : "float"                       { $$ = $1; }
    | "integer"                     { $$ = static_cast<float>($1); }
    ;


%%

namespace Tempest
{
namespace ObjMtlLoader
{
void Parser::error(const Parser::location_type& l, const std::string& m)
{
    driver.error(ToLocation(l), m);
}
}
}
