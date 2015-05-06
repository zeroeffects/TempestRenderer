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

#include "tempest/shader/shader-parser.hh"
#include "tempest/shader/shader-driver.hh"

#include "tempest/utils/logging.hh"

#include <fstream>

namespace Tempest
{
namespace Shader
{
enum BuiltInEnum
{
    /* Types */
    TGE_EFFECT_BUILTIN_FLOAT,
    TGE_EFFECT_BUILTIN_INT,
    TGE_EFFECT_BUILTIN_UINT,
    TGE_EFFECT_BUILTIN_BOOL,
    TGE_EFFECT_BUILTIN_VEC2,
    TGE_EFFECT_BUILTIN_VEC3,
    TGE_EFFECT_BUILTIN_VEC4,
    TGE_EFFECT_BUILTIN_BVEC2,
    TGE_EFFECT_BUILTIN_BVEC3,
    TGE_EFFECT_BUILTIN_BVEC4,
    TGE_EFFECT_BUILTIN_IVEC2,
    TGE_EFFECT_BUILTIN_IVEC3,
    TGE_EFFECT_BUILTIN_IVEC4,
    TGE_EFFECT_BUILTIN_UVEC2,
    TGE_EFFECT_BUILTIN_UVEC3,
    TGE_EFFECT_BUILTIN_UVEC4,
    TGE_EFFECT_BUILTIN_MAT2,
    TGE_EFFECT_BUILTIN_MAT3,
    TGE_EFFECT_BUILTIN_MAT4,
    TGE_EFFECT_BUILTIN_MAT2x2,
    TGE_EFFECT_BUILTIN_MAT2x3,
    TGE_EFFECT_BUILTIN_MAT2x4,
    TGE_EFFECT_BUILTIN_MAT3x2,
    TGE_EFFECT_BUILTIN_MAT3x3,
    TGE_EFFECT_BUILTIN_MAT3x4,
    TGE_EFFECT_BUILTIN_MAT4x2,
    TGE_EFFECT_BUILTIN_MAT4x3,
    TGE_EFFECT_BUILTIN_MAT4x4,
    TGE_EFFECT_BUILTIN_SAMPLER1D,
    TGE_EFFECT_BUILTIN_SAMPLER2D,
    TGE_EFFECT_BUILTIN_SAMPLER3D,
    TGE_EFFECT_BUILTIN_SAMPLERCUBE,
    TGE_EFFECT_BUILTIN_SAMPLER1DSHADOW,
    TGE_EFFECT_BUILTIN_SAMPLER2DSHADOW,
    TGE_EFFECT_BUILTIN_SAMPLERCUBESHADOW,
    TGE_EFFECT_BUILTIN_SAMPLER1DARRAY,
    TGE_EFFECT_BUILTIN_SAMPLER2DARRAY,
    TGE_EFFECT_BUILTIN_SAMPLER1DARRAYSHADOW,
    TGE_EFFECT_BUILTIN_SAMPLER2DARRAYSHADOW,
    TGE_EFFECT_BUILTIN_ISAMPLER1D,
    TGE_EFFECT_BUILTIN_ISAMPLER2D,
    TGE_EFFECT_BUILTIN_ISAMPLER3D,
    TGE_EFFECT_BUILTIN_ISAMPLERCUBE,
    TGE_EFFECT_BUILTIN_ISAMPLER1DARRAY,
    TGE_EFFECT_BUILTIN_ISAMPLER2DARRAY,
    TGE_EFFECT_BUILTIN_USAMPLER1D,
    TGE_EFFECT_BUILTIN_USAMPLER2D,
    TGE_EFFECT_BUILTIN_USAMPLER3D,
    TGE_EFFECT_BUILTIN_USAMPLERCUBE,
    TGE_EFFECT_BUILTIN_USAMPLER1DARRAY,
    TGE_EFFECT_BUILTIN_USAMPLER2DARRAY,
    TGE_EFFECT_BUILTIN_SAMPLER2DRECT,
    TGE_EFFECT_BUILTIN_SAMPLER2DRECTSHADOW,
    TGE_EFFECT_BUILTIN_ISAMPLER2DRECT,
    TGE_EFFECT_BUILTIN_USAMPLER2DRECT,
    TGE_EFFECT_BUILTIN_SAMPLERBUFFER,
    TGE_EFFECT_BUILTIN_ISAMPLERBUFFER,
    TGE_EFFECT_BUILTIN_USAMPLERBUFFER,
    TGE_EFFECT_BUILTIN_SAMPLER2DMS,
    TGE_EFFECT_BUILTIN_ISAMPLER2DMS,
    TGE_EFFECT_BUILTIN_USAMPLER2DMS,
    TGE_EFFECT_BUILTIN_SAMPLER2DMSARRAY,
    TGE_EFFECT_BUILTIN_ISAMPLER2DMSARRAY,
    TGE_EFFECT_BUILTIN_USAMPLER2DMSARRAY,
    TGE_EFFECT_BUILTIN_SAMPLERCUBEARRAY,
    TGE_EFFECT_BUILTIN_SAMPLERCUBEARRAYSHADOW,
    TGE_EFFECT_BUILTIN_ISAMPLERCUBEARRAY,
    TGE_EFFECT_BUILTIN_USAMPLERCUBEARRAY,
    
    /* Functions */
    /* -- Angle and Trigonometry */
    TGE_EFFECT_BUILTIN_RADIANS,
    TGE_EFFECT_BUILTIN_DEGREES,
    TGE_EFFECT_BUILTIN_SIN,
    TGE_EFFECT_BUILTIN_COS,
    TGE_EFFECT_BUILTIN_TAN,
    TGE_EFFECT_BUILTIN_ASIN,
    TGE_EFFECT_BUILTIN_ACOS,
    TGE_EFFECT_BUILTIN_ATAN,
    TGE_EFFECT_BUILTIN_SINH,
    TGE_EFFECT_BUILTIN_COSH,
    TGE_EFFECT_BUILTIN_TANH,
    TGE_EFFECT_BUILTIN_ASINH,
    TGE_EFFECT_BUILTIN_ACOSH,
    TGE_EFFECT_BUILTIN_ATANH,
    /* -- Exponential Functions */
    TGE_EFFECT_BUILTIN_POW,
    TGE_EFFECT_BUILTIN_EXP,
    TGE_EFFECT_BUILTIN_LOG,
    TGE_EFFECT_BUILTIN_EXP2,
    TGE_EFFECT_BUILTIN_LOG2,
    TGE_EFFECT_BUILTIN_SQRT,
    TGE_EFFECT_BUILTIN_INVERSESQRT,
    /* -- Common Functions */
    TGE_EFFECT_BUILTIN_ABS,
    TGE_EFFECT_BUILTIN_SIGN,
    TGE_EFFECT_BUILTIN_FLOOR,
    TGE_EFFECT_BUILTIN_TRUNC,
    TGE_EFFECT_BUILTIN_ROUND,
    TGE_EFFECT_BUILTIN_ROUNDEVEN,
    TGE_EFFECT_BUILTIN_CEIL,
    TGE_EFFECT_BUILTIN_FRACT,
    TGE_EFFECT_BUILTIN_MOD,
    TGE_EFFECT_BUILTIN_MODF,
    TGE_EFFECT_BUILTIN_MIN,
    TGE_EFFECT_BUILTIN_MAX,
    TGE_EFFECT_BUILTIN_CLAMP,
    TGE_EFFECT_BUILTIN_MIX,
    TGE_EFFECT_BUILTIN_STEP,
    TGE_EFFECT_BUILTIN_SMOOTHSTEP,
    TGE_EFFECT_BUILTIN_ISNAN,
    TGE_EFFECT_BUILTIN_ISINF,
    TGE_EFFECT_BUILTIN_FMA,
    TGE_EFFECT_BUILTIN_FREXP,
    TGE_EFFECT_BUILTIN_LDEXP,
    /* -- Packing and Unpacking Functions */
    TGE_EFFECT_BUILTIN_PACKUNORM2X16,
    TGE_EFFECT_BUILTIN_PACKUNORM4X8,
    TGE_EFFECT_BUILTIN_PACKSNORM4X8,
    TGE_EFFECT_BUILTIN_UNPACKUNORM2X16,
    TGE_EFFECT_BUILTIN_UNPACKUNORM4X8,
    TGE_EFFECT_BUILTIN_UNPACKSNORM4X8,
    /* -- Geometric Functions */
    TGE_EFFECT_BUILTIN_LENGTH,
    TGE_EFFECT_BUILTIN_DISTANCE,
    TGE_EFFECT_BUILTIN_DOT,
    TGE_EFFECT_BUILTIN_CROSS,
    TGE_EFFECT_BUILTIN_NORMALIZE,
    TGE_EFFECT_BUILTIN_FACEFORWARD,
    TGE_EFFECT_BUILTIN_REFLECT,
    TGE_EFFECT_BUILTIN_REFRACT,
    /* -- Matrix Functions */
    TGE_EFFECT_BUILTIN_MATRIXCOMPMULT,
    TGE_EFFECT_BUILTIN_OUTERPRODUCT,
    TGE_EFFECT_BUILTIN_TRANSPOSE,
    TGE_EFFECT_BUILTIN_DETERMINANT,
    TGE_EFFECT_BUILTIN_INVERSE,
    /* -- Vector Relational Functions */
    TGE_EFFECT_BUILTIN_LESSTHAN,
    TGE_EFFECT_BUILTIN_LESSTHANEQUAL,
    TGE_EFFECT_BUILTIN_GREATERTHAN,
    TGE_EFFECT_BUILTIN_GREATERTHANEQUAL,
    TGE_EFFECT_BUILTIN_EQUAL,
    TGE_EFFECT_BUILTIN_NOTEQUAL,
    TGE_EFFECT_BUILTIN_ANY,
    TGE_EFFECT_BUILTIN_ALL,
    TGE_EFFECT_BUILTIN_NOT,
    /* -- Integer Functions */
    TGE_EFFECT_BUILTIN_UADDCARRY,
    TGE_EFFECT_BUILTIN_USUBBORROW,
    TGE_EFFECT_BUILTIN_UMULEXTENDED,
    TGE_EFFECT_BUILTIN_IMULEXTENDED,
    TGE_EFFECT_BUILTIN_BITFIELDEXTRACT,
    TGE_EFFECT_BUILTIN_BITFIELDINSERT,
    TGE_EFFECT_BUILTIN_BITFIELDREVERSE,
    TGE_EFFECT_BUILTIN_BITCOUNT,
    TGE_EFFECT_BUILTIN_FINDLSB,
    TGE_EFFECT_BUILTIN_FINDMSB,
    /* -- Texture Lookup Functions */
    TGE_EFFECT_BUILTIN_TEXTURESIZE,
    TGE_EFFECT_BUILTIN_TEXTUREQUERYLOD,
    TGE_EFFECT_BUILTIN_TEXTURE,
    TGE_EFFECT_BUILTIN_TEXTUREPROJ,
    TGE_EFFECT_BUILTIN_TEXTURELOD,
    TGE_EFFECT_BUILTIN_TEXTUREOFFSET,
    TGE_EFFECT_BUILTIN_TEXELFETCH,
    TGE_EFFECT_BUILTIN_TEXELFETCHOFFSET,
    TGE_EFFECT_BUILTIN_TEXTUREPROJOFFSET,
    TGE_EFFECT_BUILTIN_TEXTURELODOFFSET,
    TGE_EFFECT_BUILTIN_TEXTUREPROJLOD,
    TGE_EFFECT_BUILTIN_TEXTUREPROJLODOFFSET,
    TGE_EFFECT_BUILTIN_TEXTUREGRAD,
    TGE_EFFECT_BUILTIN_TEXTUREGRADOFFSET,
    TGE_EFFECT_BUILTIN_TEXTUREPROJGRAD,
    TGE_EFFECT_BUILTIN_TEXTUREPROJGRADOFFSET,
    TGE_EFFECT_BUILTIN_TEXTUREGATHER,
    TGE_EFFECT_BUILTIN_TEXTUREGATHEROFFSET,
    TGE_EFFECT_BUILTIN_TEXTUREGATHEROFFSETS,
    /* -- Noise Functions */
    TGE_EFFECT_BUILTIN_NOISE1,
    TGE_EFFECT_BUILTIN_NOISE2,
    TGE_EFFECT_BUILTIN_NOISE3,
    TGE_EFFECT_BUILTIN_NOISE4,
    // Geometry Shader Functions

    /* Variables */
    /* -- Constants */
    TGE_EFFECT_BUILTIN_GL_MAX_CLIP_DISTANCE,
    TGE_EFFECT_BUILTIN_GL_MAX_DRAW_BUFFERS,
    TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS,
    TGE_EFFECT_BUILTIN_GL_MAX_TEXTURE_IMAGE_UNITS,
    TGE_EFFECT_BUILTIN_GL_MAX_VERTEX_ATTRIBS,
    TGE_EFFECT_BUILTIN_GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS,
    TGE_EFFECT_BUILTIN_GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS,
    TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_VARYING_COMPONENTS,
    TGE_EFFECT_BUILTIN_GL_MAX_VARYING_COMPONENTS,
    TGE_EFFECT_BUILTIN_GL_MAX_VARYING_FLOATS,
    TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_OUTPUT_VERTICES,
    TGE_EFFECT_BUILTIN_GL_MAX_FRAGMENT_UNIFORM_COMPONENTS,
    TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS,
    TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_UNIFORM_COMPONENTS,
    TGE_EFFECT_BUILTIN_GL_MAX_VERTEX_UNIFORM_COMPONENTS,

    TGE_EFFECT_BUILTINS
};

enum BuiltInVertexShader
{
    TGE_EFFECT_BUILTIN_IN_TGE_DRAW_ID, // It is possible that it is defined as instance index on some platforms
    TGE_EFFECT_BUILTIN_IN_GL_VERTEX_ID,
    TGE_EFFECT_BUILTIN_IN_GL_INSTANTCE_ID,
    TGE_EFFECT_BUILTIN_OUT_GL_POSITION,
    TGE_EFFECT_BUILTIN_OUT_GL_POINT_SIZE,
    TGE_EFFECT_BUILTIN_OUT_GL_CLIP_DISTANCE,

    TGE_EFFECT_BUILTINS_VS
};

enum BuiltInFragmentShader
{
    /* -- Fragment Processing Functions */
    TGE_EFFECT_BUILTIN_DFDX,
    TGE_EFFECT_BUILTIN_DFDY,
    TGE_EFFECT_BUILTIN_FWIDTH,
    /* --- Interpolation Functions */
    TGE_EFFECT_BUILTIN_INTERPOLATEATCENTROID,
    TGE_EFFECT_BUILTIN_INTERPOLATEATSAMPLE,
    TGE_EFFECT_BUILTIN_INTERPOLATEATOFFSET,
    
    /* -- Built-in Input and Output Vertex Attributes */
    TGE_EFFECT_BUILTIN_IN_GL_FRAG_COORD,
    TGE_EFFECT_BUILTINS_CONSTANT_FS = TGE_EFFECT_BUILTIN_IN_GL_FRAG_COORD,
    TGE_EFFECT_BUILTIN_IN_GL_FRONT_FACING,
    TGE_EFFECT_BUILTIN_IN_GL_CLIP_DISTANCE,
    TGE_EFFECT_BUILTIN_IN_GL_POINT_COORD,
    TGE_EFFECT_BUILTIN_IN_GL_PRIMITIVE_ID,
    TGE_EFFECT_BUILTIN_IN_GL_SAMPLE_ID,
    TGE_EFFECT_BUILTIN_IN_GL_SAMPLE_POSITION,
    TGE_EFFECT_BUILTIN_OUT_GL_FRAG_DEPTH,
    TGE_EFFECT_BUILTIN_OUT_GL_SAMPLER_MASK,

    TGE_EFFECT_BUILTINS_FS
};

struct GenType;
struct GenIType;
struct GenUType;
struct GenBType;
struct Mat;
struct Vec;
struct IVec;
struct UVec;
struct BVec;
struct GVec4;
struct vec2;
struct vec3;
struct vec4;
struct ivec2;
struct ivec3;
struct ivec4;
struct uvec2;
struct uvec3;
struct uvec4;
struct mat2x2;
struct mat2x3;
struct mat2x4;
struct mat3x2;
struct mat3x3;
struct mat3x4;
struct mat4x2;
struct mat4x3;
struct mat4x4;
typedef mat2x2 mat2;
typedef mat3x3 mat3;
typedef mat4x4 mat4;
struct GSampler1D;
struct GSampler2D;
struct GSampler3D;
struct GSamplerCube;
struct GSampler2DRect;
struct GSampler1DArray;
struct GSampler2DArray;
struct GSamplerBuffer;
struct GSampler2DMS;
struct GSampler2DMSArray;
struct GSamplerCubeArray;
struct sampler1D;
struct sampler2D;
struct sampler3D;
struct samplerCube;
struct sampler1DShadow;
struct sampler2DShadow;
struct samplerCubeShadow;
struct sampler1DArray;
struct sampler2DArray;
struct sampler1DArrayShadow;
struct sampler2DArrayShadow;
struct isampler1D;
struct isampler2D;
struct isampler3D;
struct isamplerCube;
struct isampler1DArray;
struct isampler2DArray;
struct usampler1D;
struct usampler2D;
struct usampler3D;
struct usamplerCube;
struct usampler1DArray;
struct usampler2DArray;
struct sampler2DRect;
struct sampler2DRectShadow;
struct isampler2DRect;
struct usampler2DRect;
struct samplerBuffer;
struct isamplerBuffer;
struct usamplerBuffer;
struct sampler2DMS;
struct isampler2DMS;
struct usampler2DMS;
struct sampler2DMSArray;
struct isampler2DMSArray;
struct usampler2DMSArray;
struct samplerCubeArray;
struct samplerCubeArrayShadow;
struct isamplerCubeArray;
struct usamplerCubeArray;

struct ShaderBlockType;

template<class T, size_t idx>
struct GeneratorType
{
    static const bool valid_element = false;
};

#define GENERATE_GTYPE_IDX(TBase, TEnum, idx) \
    template<> struct GeneratorType<TBase, idx> { static const size_t type_enum = TEnum; static const bool valid_element = true; };

#define GENERATE_GTYPE3(TBase, T0, T1, T2) \
    GENERATE_GTYPE_IDX(TBase, T0, 0) \
    GENERATE_GTYPE_IDX(TBase, T1, 1) \
    GENERATE_GTYPE_IDX(TBase, T2, 2)

#define GENERATE_GTYPE4(TBase, T0, T1, T2, T3) \
    GENERATE_GTYPE_IDX(TBase, T0, 0) \
    GENERATE_GTYPE_IDX(TBase, T1, 1) \
    GENERATE_GTYPE_IDX(TBase, T2, 2) \
    GENERATE_GTYPE_IDX(TBase, T3, 3)

#define GENERATE_GTYPE(TBase, T0) \
    template<size_t idx> struct GeneratorType<TBase, idx> { static const size_t type_enum = T0; static const bool valid_element = false; };

GENERATE_GTYPE(float, TGE_EFFECT_BUILTIN_FLOAT)
GENERATE_GTYPE(int, TGE_EFFECT_BUILTIN_INT)
GENERATE_GTYPE(unsigned, TGE_EFFECT_BUILTIN_UINT)
GENERATE_GTYPE(bool, TGE_EFFECT_BUILTIN_BOOL)
GENERATE_GTYPE(vec2, TGE_EFFECT_BUILTIN_VEC2)
GENERATE_GTYPE(vec3, TGE_EFFECT_BUILTIN_VEC3)
GENERATE_GTYPE(vec4, TGE_EFFECT_BUILTIN_VEC4)
GENERATE_GTYPE(ivec2, TGE_EFFECT_BUILTIN_IVEC2)
GENERATE_GTYPE(ivec3, TGE_EFFECT_BUILTIN_IVEC3)
GENERATE_GTYPE(ivec4, TGE_EFFECT_BUILTIN_IVEC4)
GENERATE_GTYPE(uvec2, TGE_EFFECT_BUILTIN_UVEC2)
GENERATE_GTYPE(uvec3, TGE_EFFECT_BUILTIN_UVEC3)
GENERATE_GTYPE(uvec4, TGE_EFFECT_BUILTIN_UVEC4)
GENERATE_GTYPE(mat2x2, TGE_EFFECT_BUILTIN_MAT2x2)
GENERATE_GTYPE(mat3x3, TGE_EFFECT_BUILTIN_MAT3x3)
GENERATE_GTYPE(mat4x4, TGE_EFFECT_BUILTIN_MAT4x4)
GENERATE_GTYPE(mat2x3, TGE_EFFECT_BUILTIN_MAT2x3)
GENERATE_GTYPE(mat2x4, TGE_EFFECT_BUILTIN_MAT2x4)
GENERATE_GTYPE(mat3x2, TGE_EFFECT_BUILTIN_MAT3x2)
GENERATE_GTYPE(mat3x4, TGE_EFFECT_BUILTIN_MAT3x4)
GENERATE_GTYPE(mat4x2, TGE_EFFECT_BUILTIN_MAT4x2)
GENERATE_GTYPE(mat4x3, TGE_EFFECT_BUILTIN_MAT4x3)
GENERATE_GTYPE(sampler1D, TGE_EFFECT_BUILTIN_SAMPLER1D)
GENERATE_GTYPE(sampler2D, TGE_EFFECT_BUILTIN_SAMPLER2D)
GENERATE_GTYPE(sampler3D, TGE_EFFECT_BUILTIN_SAMPLER3D)
GENERATE_GTYPE(samplerCube, TGE_EFFECT_BUILTIN_SAMPLERCUBE)
GENERATE_GTYPE(sampler1DShadow, TGE_EFFECT_BUILTIN_SAMPLER1DSHADOW)
GENERATE_GTYPE(sampler2DShadow, TGE_EFFECT_BUILTIN_SAMPLER2DSHADOW)
GENERATE_GTYPE(samplerCubeShadow, TGE_EFFECT_BUILTIN_SAMPLERCUBESHADOW)
GENERATE_GTYPE(sampler1DArray, TGE_EFFECT_BUILTIN_SAMPLER1DARRAY)
GENERATE_GTYPE(sampler2DArray, TGE_EFFECT_BUILTIN_SAMPLER2DARRAY)
GENERATE_GTYPE(sampler1DArrayShadow, TGE_EFFECT_BUILTIN_SAMPLER1DARRAYSHADOW)
GENERATE_GTYPE(sampler2DArrayShadow, TGE_EFFECT_BUILTIN_SAMPLER2DARRAYSHADOW)
GENERATE_GTYPE(isampler1D, TGE_EFFECT_BUILTIN_ISAMPLER1D)
GENERATE_GTYPE(isampler2D, TGE_EFFECT_BUILTIN_ISAMPLER2D)
GENERATE_GTYPE(isampler3D, TGE_EFFECT_BUILTIN_ISAMPLER3D)
GENERATE_GTYPE(isamplerCube, TGE_EFFECT_BUILTIN_ISAMPLERCUBE)
GENERATE_GTYPE(isampler1DArray, TGE_EFFECT_BUILTIN_ISAMPLER1DARRAY)
GENERATE_GTYPE(isampler2DArray, TGE_EFFECT_BUILTIN_ISAMPLER2DARRAY)
GENERATE_GTYPE(usampler1D, TGE_EFFECT_BUILTIN_USAMPLER1D)
GENERATE_GTYPE(usampler2D, TGE_EFFECT_BUILTIN_USAMPLER2D)
GENERATE_GTYPE(usampler3D, TGE_EFFECT_BUILTIN_USAMPLER3D)
GENERATE_GTYPE(usamplerCube, TGE_EFFECT_BUILTIN_USAMPLERCUBE)
GENERATE_GTYPE(usampler1DArray, TGE_EFFECT_BUILTIN_USAMPLER1DARRAY)
GENERATE_GTYPE(usampler2DArray, TGE_EFFECT_BUILTIN_USAMPLER2DARRAY)
GENERATE_GTYPE(sampler2DRect, TGE_EFFECT_BUILTIN_SAMPLER2DRECT)
GENERATE_GTYPE(sampler2DRectShadow, TGE_EFFECT_BUILTIN_SAMPLER2DRECTSHADOW)
GENERATE_GTYPE(isampler2DRect, TGE_EFFECT_BUILTIN_ISAMPLER2DRECT)
GENERATE_GTYPE(usampler2DRect, TGE_EFFECT_BUILTIN_USAMPLER2DRECT)
GENERATE_GTYPE(samplerBuffer, TGE_EFFECT_BUILTIN_SAMPLERBUFFER)
GENERATE_GTYPE(isamplerBuffer, TGE_EFFECT_BUILTIN_ISAMPLERBUFFER)
GENERATE_GTYPE(usamplerBuffer, TGE_EFFECT_BUILTIN_USAMPLERBUFFER)
GENERATE_GTYPE(sampler2DMS, TGE_EFFECT_BUILTIN_SAMPLER2DMS)
GENERATE_GTYPE(isampler2DMS, TGE_EFFECT_BUILTIN_ISAMPLER2DMS)
GENERATE_GTYPE(usampler2DMS, TGE_EFFECT_BUILTIN_USAMPLER2DMS)
GENERATE_GTYPE(sampler2DMSArray, TGE_EFFECT_BUILTIN_SAMPLER2DMSARRAY)
GENERATE_GTYPE(isampler2DMSArray, TGE_EFFECT_BUILTIN_ISAMPLER2DMSARRAY)
GENERATE_GTYPE(usampler2DMSArray, TGE_EFFECT_BUILTIN_USAMPLER2DMSARRAY)
GENERATE_GTYPE(samplerCubeArray, TGE_EFFECT_BUILTIN_SAMPLERCUBEARRAY)
GENERATE_GTYPE(samplerCubeArrayShadow, TGE_EFFECT_BUILTIN_SAMPLERCUBEARRAYSHADOW)
GENERATE_GTYPE(isamplerCubeArray, TGE_EFFECT_BUILTIN_ISAMPLERCUBEARRAY)
GENERATE_GTYPE(usamplerCubeArray, TGE_EFFECT_BUILTIN_USAMPLERCUBEARRAY)

GENERATE_GTYPE(ShaderBlockType, TGE_EFFECT_BUILTIN_SHADER_TYPE)

GENERATE_GTYPE3(GSampler1D, TGE_EFFECT_BUILTIN_SAMPLER1D, TGE_EFFECT_BUILTIN_ISAMPLER1D, TGE_EFFECT_BUILTIN_USAMPLER1D)
GENERATE_GTYPE3(GSampler2D, TGE_EFFECT_BUILTIN_SAMPLER2D, TGE_EFFECT_BUILTIN_ISAMPLER2D, TGE_EFFECT_BUILTIN_USAMPLER2D)
GENERATE_GTYPE3(GSampler3D, TGE_EFFECT_BUILTIN_SAMPLER3D, TGE_EFFECT_BUILTIN_ISAMPLER3D, TGE_EFFECT_BUILTIN_USAMPLER3D)
GENERATE_GTYPE3(GSamplerCube, TGE_EFFECT_BUILTIN_SAMPLERCUBE, TGE_EFFECT_BUILTIN_ISAMPLERCUBE, TGE_EFFECT_BUILTIN_USAMPLERCUBE)
GENERATE_GTYPE3(GSampler2DRect, TGE_EFFECT_BUILTIN_SAMPLER2DRECT, TGE_EFFECT_BUILTIN_ISAMPLER2DRECT, TGE_EFFECT_BUILTIN_USAMPLER2DRECT)
GENERATE_GTYPE3(GSampler1DArray, TGE_EFFECT_BUILTIN_SAMPLER1DARRAY, TGE_EFFECT_BUILTIN_ISAMPLER1DARRAY, TGE_EFFECT_BUILTIN_USAMPLER1DARRAY)
GENERATE_GTYPE3(GSampler2DArray, TGE_EFFECT_BUILTIN_SAMPLER2DARRAY, TGE_EFFECT_BUILTIN_ISAMPLER2DARRAY, TGE_EFFECT_BUILTIN_USAMPLER2DARRAY)
GENERATE_GTYPE3(GSamplerBuffer, TGE_EFFECT_BUILTIN_SAMPLERBUFFER, TGE_EFFECT_BUILTIN_ISAMPLERBUFFER, TGE_EFFECT_BUILTIN_USAMPLERBUFFER)
GENERATE_GTYPE3(GSampler2DMS, TGE_EFFECT_BUILTIN_SAMPLER2DMS, TGE_EFFECT_BUILTIN_ISAMPLER2DMS, TGE_EFFECT_BUILTIN_USAMPLER2DMS)
GENERATE_GTYPE3(GSampler2DMSArray, TGE_EFFECT_BUILTIN_SAMPLER2DMSARRAY, TGE_EFFECT_BUILTIN_ISAMPLER2DMSARRAY, TGE_EFFECT_BUILTIN_USAMPLER2DMSARRAY)
GENERATE_GTYPE3(GSamplerCubeArray, TGE_EFFECT_BUILTIN_SAMPLERCUBEARRAY, TGE_EFFECT_BUILTIN_ISAMPLERCUBEARRAY, TGE_EFFECT_BUILTIN_USAMPLERCUBEARRAY)

GENERATE_GTYPE4(GenType, TGE_EFFECT_BUILTIN_FLOAT, TGE_EFFECT_BUILTIN_VEC2, TGE_EFFECT_BUILTIN_VEC3, TGE_EFFECT_BUILTIN_VEC4)
GENERATE_GTYPE4(GenIType, TGE_EFFECT_BUILTIN_INT, TGE_EFFECT_BUILTIN_IVEC2, TGE_EFFECT_BUILTIN_IVEC3, TGE_EFFECT_BUILTIN_IVEC4)
GENERATE_GTYPE4(GenUType, TGE_EFFECT_BUILTIN_UINT, TGE_EFFECT_BUILTIN_UVEC2, TGE_EFFECT_BUILTIN_UVEC3, TGE_EFFECT_BUILTIN_UVEC4)
GENERATE_GTYPE4(GenBType, TGE_EFFECT_BUILTIN_BOOL, TGE_EFFECT_BUILTIN_BVEC2, TGE_EFFECT_BUILTIN_BVEC3, TGE_EFFECT_BUILTIN_BVEC4)

GENERATE_GTYPE3(Vec, TGE_EFFECT_BUILTIN_VEC2, TGE_EFFECT_BUILTIN_VEC3, TGE_EFFECT_BUILTIN_VEC4)
GENERATE_GTYPE3(GVec4, TGE_EFFECT_BUILTIN_VEC4, TGE_EFFECT_BUILTIN_IVEC4, TGE_EFFECT_BUILTIN_UVEC4)
GENERATE_GTYPE3(IVec, TGE_EFFECT_BUILTIN_IVEC2, TGE_EFFECT_BUILTIN_IVEC3, TGE_EFFECT_BUILTIN_IVEC4)
GENERATE_GTYPE3(UVec, TGE_EFFECT_BUILTIN_UVEC2, TGE_EFFECT_BUILTIN_UVEC3, TGE_EFFECT_BUILTIN_UVEC4)
GENERATE_GTYPE3(BVec, TGE_EFFECT_BUILTIN_BVEC2, TGE_EFFECT_BUILTIN_BVEC3, TGE_EFFECT_BUILTIN_BVEC4)

GENERATE_GTYPE_IDX(Mat, TGE_EFFECT_BUILTIN_MAT2x2, 0)
GENERATE_GTYPE_IDX(Mat, TGE_EFFECT_BUILTIN_MAT2x3, 1)
GENERATE_GTYPE_IDX(Mat, TGE_EFFECT_BUILTIN_MAT2x4, 2)
GENERATE_GTYPE_IDX(Mat, TGE_EFFECT_BUILTIN_MAT3x2, 3)
GENERATE_GTYPE_IDX(Mat, TGE_EFFECT_BUILTIN_MAT3x3, 4)
GENERATE_GTYPE_IDX(Mat, TGE_EFFECT_BUILTIN_MAT3x4, 5)
GENERATE_GTYPE_IDX(Mat, TGE_EFFECT_BUILTIN_MAT4x2, 6)
GENERATE_GTYPE_IDX(Mat, TGE_EFFECT_BUILTIN_MAT4x3, 7)
GENERATE_GTYPE_IDX(Mat, TGE_EFFECT_BUILTIN_MAT4x4, 8)

/*! \brief Gets the N-th type associated with the generic type used for
 *         specifying a function set that must be generated.
 *  \tparam T   the generic type.
 *  \tparam idx the index of the type associated with the generic type.
 */
template<class T, size_t idx>
struct GetType
{
    /*! \brief Gets a pointer to the type with specified index.
     *  \param _stack   the stack object which contains the built-in types.
     */
    inline static const Type* pointer(AST::ObjectPoolType& obj_pool, StackType& _stack)
    {
        auto& _node = obj_pool[_stack[GeneratorType<T, idx>::type_enum]];
        return _node.template extract<Type>();
    }
};

template<class T> struct _Arr;

/* \brief Wraps array types.
 * 
 * The main reason for this wrapping is because the type gets converted to pointer
 * which makes impossible a more "elegant" solution.
 * 
 * \tparam T    the type of the array.
 * \tparam size the size of the array.
 */
template<class T, unsigned size>
struct _Arr<T[size]>
{
    //! The array base type.
    typedef T original_type;
    
    /*! \brief In this case it does not applies properties, but changes the type to array type.
     *
     *  \param var      a pointer to the variable which contains information about the function argument.
     */
    inline static void apply(AST::ObjectPoolType& obj_pool, AST::StackType& _stack, Variable* var)
    {
        size_t pool_idx = obj_pool.size();
        obj_pool.push_back(CreateTypeNode<ArrayType>(var->getType(), CreateNode<Value<unsigned>>(TGE_DEFAULT_LOCATION, size)));
        _stack.push_back(pool_idx);
        var->setType(obj_pool[_stack.back()].extract<Type>());
    }
};

/*! \brief Wraps the input qualifier.
 *  
 *  \tparam T the base type.
 */
template<class T>
struct _In
{
    //! The base type.
    typedef T original_type;
    
    /*! \brief Applies storage attribute.
     * 
     *  \param var      a pointer to the variable which contains information about the function argument.
     */ 
    inline static void apply(AST::ObjectPoolType& obj_pool, AST::StackType& _stack, Variable* var)
    {
        var->setStorage(StorageQualifier::In);
    }
};

/*! \brief Wraps the output qualifier.
 *  
 *  \tparam T the base type.
 */
template<class T>
struct _Out
{
    //! The base type.
    typedef T original_type;
    
    /*! \brief Applies storage attribute.
     * 
     *  \param var      a pointer to the variable which contains information about the function argument.
     */ 
    inline static void apply(AST::ObjectPoolType& obj_pool, AST::StackType& _stack, Variable* var)
    {
        var->setStorage(StorageQualifier::Out);
    }
};

/*! \brief The base template for stripping attributes, which in this case does
 *         not strip anything.
 * 
 *  \tparam T   the final base type.
 */
template<class T>
struct Parameters
{
    //! The base type.
    typedef T original_type;
    
    //! Empty because the specified type is not composite.
    inline static void apply(AST::ObjectPoolType& obj_pool, AST::StackType& _stack, Variable* var) {}
};

/*! \brief Template used for applying attributes.
 * 
 *  \tparam TStripped    the attribute that was stripped on this pass.
 *  \tparam TBase        the base type which could be also a composite type.
 */
template<template<class TBase> class TStripped, class TBase>
struct Parameters<TStripped<TBase>>
{
    //! The base type.
    typedef typename TStripped<TBase>::original_type original_type;
    
    /*! \brief Strips the attributes and applies them. 
     */
    inline static void apply(AST::ObjectPoolType& obj_pool, AST::StackType& _stack, Variable* var)
    {
        TStripped<TBase>::apply(obj_pool, _stack, var);
        Parameters<TBase>::apply(obj_pool, _stack, var);
    }
};

/*! \brief Creates a function argument declaration.
 * 
 *  \param T    the type of the function argument.
 *  \param idx  the index of the type associated with generator type.
 */
template<class T, size_t idx>
struct CreateDeclaration
{
    //! Generates the function argument declaration based on the specified template arguments.
    inline static NodeT<Declaration> pointer(AST::ObjectPoolType& obj_pool, AST::StackType& _stack)
    {
        NodeT<Variable> var = CreateNodeTyped<Variable>(TGE_DEFAULT_LOCATION, GetType<typename Parameters<T>::original_type, idx>::pointer(obj_pool, _stack), "");
        Parameters<T>::apply(obj_pool, _stack, var.get());
        return CreateNodeTyped<Declaration>(TGE_DEFAULT_LOCATION, std::move(var));
    }
};

/*! \brief Generates the void type.
 */
template<size_t idx>
struct GetType<void, idx>
{
    /*! \brief Returns the void type.
     *  \param _stack   the stack object which contains the built-in types.
     */
    inline static const Type* pointer(AST::ObjectPoolType& obj_pool, AST::StackType& _stack)
    {
        return nullptr;
    }
};

/*! \brief Generates the list of arguments from the specified template arguments
 *         list.
 *
 *  \tparam idx     the index of the currently generated element inside the
 *                  function set.
 *  \tparam T
 *  \tparam TArgs   
 */
template<size_t idx, class T, class... TArgs>
struct GenList
{
    inline static NodeT<List> generate(AST::ObjectPoolType& obj_pool, AST::StackType& _stack)
    {
        return CreateNodeTyped<ListElement>(TGE_DEFAULT_LOCATION, ListType::SemicolonSeparated, CreateDeclaration<T, idx>::pointer(obj_pool, _stack),
                                            GenList<idx, TArgs...>::generate(obj_pool, _stack));
    }
};

/*! \brief Sets the end of the generated arguments list.
 *
 *  \tparam idx     the index of the currently generated element inside the
 *                  function set.
 *  \tparam T
 */
template<size_t idx, class T>
struct GenList<idx, T>
{
    inline static NodeT<List> generate(AST::ObjectPoolType& obj_pool, AST::StackType& _stack)
    {
        return CreateNodeTyped<ListElement>(TGE_DEFAULT_LOCATION, ListType::SemicolonSeparated, CreateDeclaration<T, idx>::pointer(obj_pool, _stack),
                                            NodeT<List>());
    }
};

/*! \brief A template class which generates the N-th function of the function
 *         set.
 *
 *  \tparam idx     the index of the currently generated element inside the
 *                  function set.
 *  \tparam TRet    the return type from the prototype used for generating the
 *                  function set.
 *  \tparam TArgs
 */
template<size_t idx, class TRet, class... TArgs>
struct GenFunction
{
    /*! \brief Generates the function with the specified index in the function
     *         set.
     *  The actual index is used for iterating through the types associated
     *  with the prototype.
     *  \param _stack       the stack object which includes the types used during
     *                      the generation of the functions with the specified
     *                      prototype.
     * \param func_set      the function set that must be populated.
     */
    inline static void generate(AST::ObjectPoolType& obj_pool, AST::StackType& _stack, FunctionSet* func_set)
    {
        auto var_list = GenList<idx, TArgs...>::generate(obj_pool, _stack);
        auto func = CreateNodeTyped<FunctionDeclaration>(TGE_DEFAULT_LOCATION, GetType<TRet, idx>::pointer(obj_pool, _stack), func_set->getNodeName(), std::move(var_list));
        func_set->pushFunction(std::move(func));
    };
};

template<size_t idx, class... TArgs> struct ValidGeneratorType;

template<size_t idx, class T, class... TArgs>
struct ValidGeneratorType<idx, T, TArgs...>
{
    static const bool value = GeneratorType<T, idx>::valid_element | ValidGeneratorType<idx, TArgs...>::value;
};

template<size_t idx, class T>
struct ValidGeneratorType<idx, T>
{
    static const bool value = GeneratorType<T, idx>::valid_element;
};

/*! \brief A template class which generates the N-th function of the function
 *         set and executes the operation for the next function that must be
 *         generated.
 *
 *  \tparam not_end indicates whether it is the end of the function set
 *                  (assumed 'true' for this specialization of the template).
 *  \tparam idx     the index of the currently generated element inside the
 *                  function set.
 *  \tparam TRet    the return type from the prototype used for generating the
 *                  function set.
 *  \tparam TArgs   the types of the arguments to the functions that are going
 *                  to be generated.
 */
template<bool not_end, size_t idx, class TRet, class... TArgs>
struct GenFSet
{
    /*! \brief Generates the function with the specified index in the function
     *         set.
     *  The actual index is used for iterating through the types associated
     *  with the prototype.
     *  \param _stack       the stack object which includes the types used during
     *                      the generation of the functions with the specified
     *                      prototype.
     * \param func_set      the function set that must be populated.
     */
    inline static void generate(AST::ObjectPoolType& obj_pool, AST::StackType& _stack, FunctionSet* func_set)
    {
        GenFunction<idx, TRet, TArgs...>::generate(obj_pool, _stack, func_set);
        const size_t nidx = idx+1;
        // determines whether there are still types associated with specified
        // ones in the profile and executes the right specialization of the
        // template used for generating the function set.
        GenFSet<ValidGeneratorType<nidx, TRet, TArgs...>::value, nidx, TRet, TArgs...>::generate(obj_pool, _stack, func_set);
    }
};

/*! \brief A template class representing the end of the generated function set.
 *
 *  \tparam idx     the index of the currently generated element inside the
 *                  function set.
 *  \tparam TRet    the return type from the prototype used for generating the
 *                  function set.
 *  \tparam TArgs   the types of the arguments to the functions that are going
 *                  to be generated.
 */
template<size_t idx, class TRet, class... TArgs>
struct GenFSet<false, idx, TRet, TArgs...>
{
    /*! \brief Does not generate anything because all functions associated with the
     *         specified prototype were generated.
     */
    inline static void generate(AST::ObjectPoolType& obj_pool, AST::StackType& _stack, FunctionSet* func_set) {}
};

template<class T> struct BuiltInFunction;

/*! \brief Used for populating a function set with functions with the specified
 *         arguments.
 *
 *  It wraps GenFSet with a more convenient interface.
 *  \tparam TRet        the return type of the functions that are going to be
 *                      generated to populate the function set.
 *  \tparam TArgs       the types of the arguments to the functions that are going
 *                      to be generated.
 */
template<class TRet, class... TArgs>
struct BuiltInFunction<TRet(TArgs...)>
{
    /*! \brief Adds the functions with the specified prototype to the function
     *         set.
     *  \param _stack   the stack object which includes the types used during
     *                  the generation of the functions with the specified
     *                  prototype.
     *  \param elem     the function set that must be populated.
     */
    inline static void addTo(AST::ObjectPoolType& obj_pool, AST::StackType& _stack, FunctionSet* elem)
    {
        GenFSet<true, 0, TRet, TArgs...>::generate(obj_pool, _stack, elem);
    }
};

Driver::Driver()
{
// Common Shader Built-ins
    m_ObjectPool.reserve(TGE_EFFECT_BUILTINS);
    m_ShaderBuiltIns.resize(TGE_EFFECT_BUILTINS);

    // Types
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_FLOAT] = createBuiltInType<ScalarType>(false, "float");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_INT] = createBuiltInType<ScalarType>(true, "int");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_UINT] = createBuiltInType<ScalarType>(true, "uint");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_BOOL] = createBuiltInType<ScalarType>(false, "bool");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_VEC2] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_FLOAT), 2, "vec2");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_VEC3] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_FLOAT), 3, "vec3");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_VEC4] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_FLOAT), 4, "vec4");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_BVEC2] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_BOOL), 2, "bvec2");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_BVEC3] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_BOOL), 3, "bvec3");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_BVEC4] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_BOOL), 4, "bvec4");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_IVEC2] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), 2, "ivec2");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_IVEC3] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), 3, "ivec3");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_IVEC4] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), 4, "ivec4");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_UVEC2] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_UINT), 2, "uvec2");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_UVEC3] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_UINT), 3, "uvec3");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_UVEC4] = createBuiltInType<VectorType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_UINT), 4, "uvec4");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT2x2] = createBuiltInType<MatrixType>(2, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC2), "mat2x2");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT2x3] = createBuiltInType<MatrixType>(2, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC3), "mat2x3");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT2x4] = createBuiltInType<MatrixType>(2, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC4), "mat2x4");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT3x2] = createBuiltInType<MatrixType>(3, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC2), "mat3x2");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT3x3] = createBuiltInType<MatrixType>(3, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC3), "mat3x3");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT3x4] = createBuiltInType<MatrixType>(3, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC4), "mat3x4");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT4x2] = createBuiltInType<MatrixType>(4, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC2), "mat4x2");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT4x3] = createBuiltInType<MatrixType>(4, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC3), "mat4x3");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT4x4] = createBuiltInType<MatrixType>(4, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC4), "mat4x4");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT2] = createBuiltInNode<Typedef>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_MAT2x2), "mat2");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT3] = createBuiltInNode<Typedef>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_MAT3x3), "mat3");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAT4] = createBuiltInNode<Typedef>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_MAT4x4), "mat4");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER1D] = createBuiltInType<SamplerType>("sampler1D");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER2D] = createBuiltInType<SamplerType>("sampler2D");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER3D] = createBuiltInType<SamplerType>("sampler3D");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLERCUBE] = createBuiltInType<SamplerType>("samplerCube");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER1DSHADOW] = createBuiltInType<SamplerType>("sampler1DShadow");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER2DSHADOW] = createBuiltInType<SamplerType>("sampler2DShadow");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLERCUBESHADOW] = createBuiltInType<SamplerType>("samplerCubeShadow");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER1DARRAY] = createBuiltInType<SamplerType>("sampler1DArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER2DARRAY] = createBuiltInType<SamplerType>("sampler2DArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER1DARRAYSHADOW] = createBuiltInType<SamplerType>("sampler1DArrayShadow");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER2DARRAYSHADOW] = createBuiltInType<SamplerType>("sampler2DArrayShadow");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLER1D] = createBuiltInType<SamplerType>("isampler1D");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLER2D] = createBuiltInType<SamplerType>("isampler2D");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLER3D] = createBuiltInType<SamplerType>("isampler3D");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLERCUBE] = createBuiltInType<SamplerType>("isamplerCube");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLER1DARRAY] = createBuiltInType<SamplerType>("isampler1DArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLER2DARRAY] = createBuiltInType<SamplerType>("isampler2DArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLER1D] = createBuiltInType<SamplerType>("usampler1D");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLER2D] = createBuiltInType<SamplerType>("usampler2D");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLER3D] = createBuiltInType<SamplerType>("usampler3D");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLERCUBE] = createBuiltInType<SamplerType>("usamplerCube");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLER1DARRAY] = createBuiltInType<SamplerType>("usampler1DArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLER2DARRAY] = createBuiltInType<SamplerType>("usampler2DArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER2DRECT] = createBuiltInType<SamplerType>("sampler2DRect");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER2DRECTSHADOW] = createBuiltInType<SamplerType>("sampler2DRectShadow");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLER2DRECT] = createBuiltInType<SamplerType>("isampler2DRect");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLER2DRECT] = createBuiltInType<SamplerType>("usampler2DRect");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLERBUFFER] = createBuiltInType<SamplerType>("samplerBuffer");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLERBUFFER] = createBuiltInType<SamplerType>("isamplerBuffer");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLERBUFFER] = createBuiltInType<SamplerType>("usamplerBuffer");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER2DMS] = createBuiltInType<SamplerType>("sampler2DMS");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLER2DMS] = createBuiltInType<SamplerType>("isampler2DMS");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLER2DMS] = createBuiltInType<SamplerType>("usampler2DMS");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLER2DMSARRAY] = createBuiltInType<SamplerType>("sampler2DMSArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLER2DMSARRAY] = createBuiltInType<SamplerType>("isampler2DMSArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLER2DMSARRAY] = createBuiltInType<SamplerType>("usampler2DMSArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLERCUBEARRAY] = createBuiltInType<SamplerType>("samplerCubeArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SAMPLERCUBEARRAYSHADOW] = createBuiltInType<SamplerType>("samplerCubeArrayShadow");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISAMPLERCUBEARRAY] = createBuiltInType<SamplerType>("isamplerCubeArray");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USAMPLERCUBEARRAY] = createBuiltInType<SamplerType>("usamplerCubeArray");

    FunctionSet* func_set;
    // Functions
    // -- Angle and Trigonometry
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_RADIANS] = createFunctionSet(&func_set, "radians");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_DEGREES] = createFunctionSet(&func_set, "degrees");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SIN] = createFunctionSet(&func_set, "sin");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_COS] = createFunctionSet(&func_set, "cos");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TAN] = createFunctionSet(&func_set, "tan");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ASIN] = createFunctionSet(&func_set, "asin");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ACOS] = createFunctionSet(&func_set, "acos");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ATAN] = createFunctionSet(&func_set, "atan");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenType (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SINH] = createFunctionSet(&func_set, "sinh");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_COSH] = createFunctionSet(&func_set, "cosh");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TANH] = createFunctionSet(&func_set, "tanh");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ASINH] = createFunctionSet(&func_set, "asinh");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ACOSH] = createFunctionSet(&func_set, "acosh");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ATANH] = createFunctionSet(&func_set, "atanh");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // -- Exponential Functions
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_POW] = createFunctionSet(&func_set, "pow");
    BuiltInFunction<GenType (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_EXP] = createFunctionSet(&func_set, "exp");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_LOG] = createFunctionSet(&func_set, "log");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_EXP2] = createFunctionSet(&func_set, "exp2");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_LOG2] = createFunctionSet(&func_set, "log2");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SQRT] = createFunctionSet(&func_set, "sqrt");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_INVERSESQRT] = createFunctionSet(&func_set, "inversesqrt");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // -- Common Functions
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ABS] = createFunctionSet(&func_set, "abs");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenIType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SIGN] = createFunctionSet(&func_set, "sign");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenIType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_FLOOR] = createFunctionSet(&func_set, "floor");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TRUNC] = createFunctionSet(&func_set, "trunc");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ROUND] = createFunctionSet(&func_set, "round");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ROUNDEVEN] = createFunctionSet(&func_set, "roundEven");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_CEIL] = createFunctionSet(&func_set, "ceil");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_FRACT] = createFunctionSet(&func_set, "fract");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MOD] = createFunctionSet(&func_set, "mod");
    BuiltInFunction<GenType (GenType, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenType (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MODF] = createFunctionSet(&func_set, "modf");
    BuiltInFunction<GenType (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MIN] = createFunctionSet(&func_set, "min");
    BuiltInFunction<GenType (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenType (GenType, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenIType, GenIType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenIType, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenUType (GenUType, GenUType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenUType (GenUType, unsigned)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MAX] = createFunctionSet(&func_set, "max");
    BuiltInFunction<GenType (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenType (GenType, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenIType, GenIType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenIType, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenUType (GenUType, GenUType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenUType (GenUType, unsigned)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_CLAMP] = createFunctionSet(&func_set, "clamp");
    BuiltInFunction<GenType (GenType, GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenType (GenType, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenIType, GenIType, GenIType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenIType, int, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenUType (GenUType, GenUType, GenUType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenUType (GenUType, unsigned, unsigned)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MIX] = createFunctionSet(&func_set, "mix");
    BuiltInFunction<GenType (GenType, GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenType (GenType, GenType, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenType (GenType, GenType, GenBType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_STEP] = createFunctionSet(&func_set, "step");
    BuiltInFunction<GenType (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenType (float, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_SMOOTHSTEP] = createFunctionSet(&func_set, "smoothstep");
    BuiltInFunction<GenType (GenType, GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenType (float, float, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISNAN] = createFunctionSet(&func_set, "isnan");
    BuiltInFunction<GenBType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ISINF] = createFunctionSet(&func_set, "isinf");
    BuiltInFunction<GenBType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_FMA] = createFunctionSet(&func_set, "fma");
    BuiltInFunction<GenType (GenType, GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_FREXP] = createFunctionSet(&func_set, "frexp");
    BuiltInFunction<GenType (GenType, _In<GenIType>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_LDEXP] = createFunctionSet(&func_set, "ldexp");
    BuiltInFunction<GenType (GenType, _In<GenIType>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // -- Packing and Unpacking Functions
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_PACKUNORM2X16] = createFunctionSet(&func_set, "packUnorm2x16");
    BuiltInFunction<unsigned (vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_PACKUNORM4X8] = createFunctionSet(&func_set, "packUnorm4x8");
    BuiltInFunction<unsigned (vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_PACKSNORM4X8] = createFunctionSet(&func_set, "packSnorm4x8");
    BuiltInFunction<unsigned (vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_UNPACKUNORM2X16] = createFunctionSet(&func_set, "unpackUnorm2x16");
    BuiltInFunction<vec2 (unsigned)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_UNPACKUNORM4X8] = createFunctionSet(&func_set, "unpackUnorm4x8");
    BuiltInFunction<vec4 (unsigned)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_UNPACKSNORM4X8]  = createFunctionSet(&func_set, "unpackSnorm4x8");
    BuiltInFunction<vec4 (unsigned)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // -- Geometric Functions
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_LENGTH] = createFunctionSet(&func_set, "length");
    BuiltInFunction<float (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_DISTANCE] = createFunctionSet(&func_set, "distance");
    BuiltInFunction<float (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_DOT] = createFunctionSet(&func_set, "dot");
    BuiltInFunction<float (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_CROSS] = createFunctionSet(&func_set, "cross");
    BuiltInFunction<vec3 (vec3, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_NORMALIZE] = createFunctionSet(&func_set, "normalize");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_FACEFORWARD] = createFunctionSet(&func_set, "faceforward");
    BuiltInFunction<GenType (GenType, GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_REFLECT] = createFunctionSet(&func_set, "reflect");
    BuiltInFunction<GenType (GenType, GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_REFRACT] = createFunctionSet(&func_set, "refract");
    BuiltInFunction<GenType (GenType, GenType, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // -- Matrix Functions
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_MATRIXCOMPMULT] = createFunctionSet(&func_set, "matrixCompMult");
    BuiltInFunction<Mat (Mat, Mat)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_OUTERPRODUCT] = createFunctionSet(&func_set, "outerProduct");
    BuiltInFunction<mat2 (vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat3 (vec3, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat4 (vec4, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat2x3 (vec3, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat3x2 (vec2, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat2x4 (vec4, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat4x2 (vec2, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat3x4 (vec4, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat4x3 (vec3, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TRANSPOSE] = createFunctionSet(&func_set, "transpose");
    BuiltInFunction<Mat (Mat)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_DETERMINANT] = createFunctionSet(&func_set, "determinant");
    BuiltInFunction<mat2 (mat2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat3 (mat3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat4 (mat4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_INVERSE] = createFunctionSet(&func_set, "inverse");
    BuiltInFunction<mat2 (mat2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat3 (mat3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<mat4 (mat4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // -- Vector Relational Functions
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_LESSTHAN] = createFunctionSet(&func_set, "lessThan");
    BuiltInFunction<BVec (Vec, Vec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (IVec, IVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (UVec, UVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_LESSTHANEQUAL] = createFunctionSet(&func_set, "lessThanEqual");
    BuiltInFunction<BVec (Vec, Vec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (IVec, IVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (UVec, UVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GREATERTHAN] = createFunctionSet(&func_set, "greaterThan");
    BuiltInFunction<BVec (Vec, Vec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (IVec, IVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (UVec, UVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GREATERTHANEQUAL] = createFunctionSet(&func_set, "greaterThanEqual");
    BuiltInFunction<BVec (Vec, Vec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (IVec, IVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (UVec, UVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_EQUAL] = createFunctionSet(&func_set, "equal");
    BuiltInFunction<BVec (Vec, Vec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (IVec, IVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (UVec, UVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (BVec, BVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_NOTEQUAL] = createFunctionSet(&func_set, "notEqual");
    BuiltInFunction<BVec (Vec, Vec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (IVec, IVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (UVec, UVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<BVec (BVec, BVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ANY] = createFunctionSet(&func_set, "any");
    BuiltInFunction<bool (BVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_ALL] = createFunctionSet(&func_set, "all");
    BuiltInFunction<bool (BVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_NOT] = createFunctionSet(&func_set, "not");
    BuiltInFunction<BVec (BVec)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // -- Integer Functions
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_UADDCARRY] = createFunctionSet(&func_set, "uaddCarry");
    BuiltInFunction<GenUType (GenUType, GenUType, _Out<GenUType>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_USUBBORROW] = createFunctionSet(&func_set, "usubBorrow");
    BuiltInFunction<GenUType (GenUType, GenUType, _Out<GenUType>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_UMULEXTENDED] = createFunctionSet(&func_set, "umulExtended");
    BuiltInFunction<void (GenUType, GenUType, _Out<GenUType>, _Out<GenUType>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_IMULEXTENDED] = createFunctionSet(&func_set, "imulExtended");
    BuiltInFunction<void (GenIType, GenIType, _Out<GenIType>, _Out<GenIType>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_BITFIELDEXTRACT] = createFunctionSet(&func_set, "bitfieldExtract");
    BuiltInFunction<GenIType (GenIType, int, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenUType (GenUType, int, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_BITFIELDINSERT] = createFunctionSet(&func_set, "bitfieldInsert");
    BuiltInFunction<GenIType (GenIType, GenIType, int, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenUType (GenUType, GenUType, int, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_BITFIELDREVERSE] = createFunctionSet(&func_set, "bitfieldReverse");
    BuiltInFunction<GenIType (GenIType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenUType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_BITCOUNT] = createFunctionSet(&func_set, "bitCount");
    BuiltInFunction<GenIType (GenIType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenUType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_FINDLSB] = createFunctionSet(&func_set, "findLSB");
    BuiltInFunction<GenIType (GenIType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenUType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_FINDMSB] = createFunctionSet(&func_set, "findMSB");
    BuiltInFunction<GenIType (GenIType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GenIType (GenUType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // -- Texture Lookup Functions
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTURESIZE] = createFunctionSet(&func_set, "textureSize");
    BuiltInFunction<int   (GSampler1D, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (GSampler2D, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec3 (GSampler3D, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (GSamplerCube, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<int   (sampler1DShadow, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (sampler2DShadow, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (samplerCubeShadow, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec3 (samplerCubeArray, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec3 (samplerCubeArrayShadow, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (GSampler2DRect)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (sampler2DRectShadow)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (GSampler1DArray, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec3 (GSampler2DArray, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (sampler1DArrayShadow, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec3 (sampler2DArrayShadow, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<int   (GSamplerBuffer)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (GSampler2DMS)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<ivec2 (GSampler2DMSArray)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREQUERYLOD] = createFunctionSet(&func_set, "textureQueryLOD");
    BuiltInFunction<vec2 (GSampler1D, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (GSampler2D, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (GSampler3D, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (GSamplerCube, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (GSampler1DArray, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (GSampler2DArray, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (GSamplerCubeArray, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (sampler1DShadow, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (sampler2DShadow, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (samplerCubeShadow, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (sampler1DArrayShadow, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (sampler2DArrayShadow, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (samplerCubeArrayShadow, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTURE] = createFunctionSet(&func_set, "texture");
    BuiltInFunction<GVec4 (GSampler1D, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCube, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCube, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (samplerCubeShadow, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (samplerCubeShadow, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DRectShadow, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, vec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCubeArray, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCubeArray, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DArrayShadow, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DArrayShadow, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DArrayShadow, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (samplerCubeArrayShadow, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREPROJ] = createFunctionSet(&func_set, "textureProj");
    BuiltInFunction<GVec4 (GSampler1D, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DRectShadow, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTURELOD] = createFunctionSet(&func_set, "textureLod");
    BuiltInFunction<GVec4 (GSampler1D, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCube, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, vec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DArrayShadow, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCubeArray, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREOFFSET] = createFunctionSet(&func_set, "textureOffset");
    BuiltInFunction<GVec4 (GSampler1D, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, float, int, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, ivec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec3, ivec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec3, ivec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DRectShadow, vec3, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec3, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec3, int, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec3, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec3, ivec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, vec2, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, vec2, int, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, ivec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DArrayShadow, vec3, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DArrayShadow, vec3, int, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXELFETCH] = createFunctionSet(&func_set, "texelFetch");
    BuiltInFunction<GVec4 (GSampler1D, int, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, ivec2, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, ivec3, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, ivec2, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, ivec3, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerBuffer, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DMS, ivec2, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DMSArray, ivec3, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXELFETCHOFFSET] = createFunctionSet(&func_set, "texelFetchOffset");
    BuiltInFunction<GVec4 (GSampler1D, int, int, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, ivec2, int, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, ivec3, int, ivec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, ivec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, ivec2, int, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, ivec3, int, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREPROJOFFSET] = createFunctionSet(&func_set, "textureProjOffset");
    BuiltInFunction<GVec4 (GSampler1D, vec2, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec2, int, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec4, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec4, int, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec3, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec3, ivec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec4, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec4, ivec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec4, ivec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec4, ivec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec3, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec4, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DRectShadow, vec4, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec4, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec4, int, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec4, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec4, ivec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTURELODOFFSET] = createFunctionSet(&func_set, "textureLodOffset");
    BuiltInFunction<GVec4 (GSampler1D, float, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, float, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec3, float, ivec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec3, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec3, float, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, vec2, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, float, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DArrayShadow, vec3, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREPROJLOD] = createFunctionSet(&func_set, "textureProjLod");
    BuiltInFunction<GVec4 (GSampler1D, vec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREPROJLODOFFSET] = createFunctionSet(&func_set, "textureProjLodOffset");
    BuiltInFunction<GVec4 (GSampler1D, vec2, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec4, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec3, float, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec4, float, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec4, float, ivec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec4, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec4, float, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREGRAD] = createFunctionSet(&func_set, "textureGrad");
    BuiltInFunction<GVec4 (GSampler1D, float, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec3, vec3, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCube, vec3, vec3, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec2, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DRectShadow, vec3, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec3, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec3, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (samplerCubeShadow, vec4, vec3, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, vec2, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DArrayShadow, vec3, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DArrayShadow, vec4, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCubeArray, vec4, vec3, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREGRADOFFSET] = createFunctionSet(&func_set, "textureGradOffset");
    BuiltInFunction<GVec4 (GSampler1D, float, float, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, vec2, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec3, vec3, vec3, ivec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec2, vec2, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DRectShadow, vec3, vec2, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec3, float, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec3, vec2, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1DArray, vec2, float, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, vec2, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DArrayShadow, vec3, float, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DArrayShadow, vec4, vec2, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREPROJGRAD] = createFunctionSet(&func_set, "textureProjGrad");
    BuiltInFunction<GVec4 (GSampler1D, vec2, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec4, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec3, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec4, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec4, vec3, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec3, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec4, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DRectShadow, vec4, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec4, float, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec4, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREPROJGRADOFFSET] = createFunctionSet(&func_set, "textureProjGradOffset");
    BuiltInFunction<GVec4 (GSampler1D, vec2, float, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler1D, vec4, float, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec3, vec2, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec4, vec2, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec3, vec2, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DRect, vec4, vec2, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DRectShadow, vec4, vec2, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler3D, vec4, vec3, vec3, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler1DShadow, vec4, float, float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<float (sampler2DShadow, vec4, vec2, vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREGATHER] = createFunctionSet(&func_set, "textureGather");
    BuiltInFunction<GVec4 (GSampler2D, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCube, vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCube, vec3, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCubeArray, vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSamplerCubeArray, vec4, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (sampler2DShadow, vec2, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (sampler2DArrayShadow, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (samplerCubeShadow, vec3, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (samplerCubeArrayShadow, vec4, float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREGATHEROFFSET] = createFunctionSet(&func_set, "textureGatherOffset");
    BuiltInFunction<GVec4 (GSampler2D, vec2, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, ivec2, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, ivec2, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (sampler2DShadow, vec2, float, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (sampler2DArrayShadow, vec3, float, ivec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_TEXTUREGATHEROFFSETS] = createFunctionSet(&func_set, "textureGatherOffsets");
    BuiltInFunction<GVec4 (GSampler2D, vec2, _Arr<ivec2[4]>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2D, vec2, _Arr<ivec2[4]>, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, _Arr<ivec2[4]>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<GVec4 (GSampler2DArray, vec3, _Arr<ivec2[4]>, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (sampler2DShadow, vec2, float, _Arr<ivec2[4]>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (sampler2DArrayShadow, vec3, float, _Arr<ivec2[4]>)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // -- Noise Functions
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_NOISE1] = createFunctionSet(&func_set, "noise1");
    BuiltInFunction<float (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_NOISE2] = createFunctionSet(&func_set, "noise2");
    BuiltInFunction<vec2 (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_NOISE3] = createFunctionSet(&func_set, "noise3");
    BuiltInFunction<vec3 (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_NOISE4] = createFunctionSet(&func_set, "noise4");
    BuiltInFunction<vec4 (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    // TODO: Geometry Shader Functions

    // Variables
    // -- Constants
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_CLIP_DISTANCE] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxClipDistance");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_DRAW_BUFFERS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxDrawBuffers");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxGeometryTextureImageUnits");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_TEXTURE_IMAGE_UNITS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxTextureImageUnits");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_VERTEX_ATTRIBS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxVertexAttribs");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxVertexTextureImageUnits");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxCombinedTextureImageUnits");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_VARYING_COMPONENTS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxGeometryVaryingComponents");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_VARYING_COMPONENTS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxVaryingComponents");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_VARYING_FLOATS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxVaryingFloats");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_OUTPUT_VERTICES] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxGeometryOutputVertices");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_FRAGMENT_UNIFORM_COMPONENTS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxFragmentUniformComponents");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxGeometryTotalOutputComponents");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_GEOMETRY_UNIFORM_COMPONENTS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxGeometryUniformComponents");
    m_ShaderBuiltIns[TGE_EFFECT_BUILTIN_GL_MAX_VERTEX_UNIFORM_COMPONENTS] = createBuiltInNode<Variable>(StorageQualifier::Const, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_MaxVertexUniformComponents");
    TGE_ASSERT(std::find_if(m_ShaderBuiltIns.begin(), m_ShaderBuiltIns.end(), [this](size_t idx){ return !this->m_ObjectPool[idx]; }) == m_ShaderBuiltIns.end(), "Unfilled built-ins");

    m_Stack.assign(m_ShaderBuiltIns.begin(), m_ShaderBuiltIns.end());

// Fragment Shader Built-ins
    m_FSBuiltIns.resize(TGE_EFFECT_BUILTINS_CONSTANT_FS);
    // -- Fragment Processing Functions
    // WARNING: It uses m_ShaderBuiltIns for the rest of the built-ins.
    m_FSBuiltIns[TGE_EFFECT_BUILTIN_DFDX] = createFunctionSet(&func_set, "dFdx");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_FSBuiltIns[TGE_EFFECT_BUILTIN_DFDY] = createFunctionSet(&func_set, "dFdy");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_FSBuiltIns[TGE_EFFECT_BUILTIN_FWIDTH] = createFunctionSet(&func_set, "fwidth");
    BuiltInFunction<GenType (GenType)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_FSBuiltIns[TGE_EFFECT_BUILTIN_INTERPOLATEATCENTROID] = createFunctionSet(&func_set, "interpolateAtCentroid");
    BuiltInFunction<float (float)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec3 (vec3)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (vec4)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_FSBuiltIns[TGE_EFFECT_BUILTIN_INTERPOLATEATSAMPLE] = createFunctionSet(&func_set, "interpolateAtSample");
    BuiltInFunction<float (float, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (vec2, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec3 (vec3, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (vec4, int)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    m_FSBuiltIns[TGE_EFFECT_BUILTIN_INTERPOLATEATOFFSET] = createFunctionSet(&func_set, "interpolateAtOffset");
    BuiltInFunction<float (float, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec2 (vec2, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec3 (vec3, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    BuiltInFunction<vec4 (vec4, vec2)>::addTo(m_ObjectPool, m_ShaderBuiltIns, func_set);
    TGE_ASSERT(std::find_if(m_FSBuiltIns.begin(), m_FSBuiltIns.end(), [this](size_t idx){ return !this->m_ObjectPool[idx]; }) == m_FSBuiltIns.end(), "Unfilled fragment shader built-ins");
}

Driver::~Driver()
{
}

void Driver::beginShader(ShaderType shader_type)
{
    TGE_ASSERT(m_StackPointers.empty(), "Expected empty shader stack");
    size_t stack_pointer = m_Stack.size();
    m_StackPointers.push_back(stack_pointer);
    switch(shader_type)
    {
    case ShaderType::VertexShader:
    {
        size_t current_size = m_Stack.size();
        m_Stack.resize(m_Stack.size() + TGE_EFFECT_BUILTINS_VS);
        StackType::iterator::pointer vs_p = &m_Stack[current_size];
        vs_p[TGE_EFFECT_BUILTIN_IN_TGE_DRAW_ID] = createBuiltInNode<Variable>(StorageQualifier::In, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "tge_DrawID");
        vs_p[TGE_EFFECT_BUILTIN_IN_GL_VERTEX_ID] = createBuiltInNode<Variable>(StorageQualifier::In, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_VertexID");
        vs_p[TGE_EFFECT_BUILTIN_IN_GL_INSTANTCE_ID] = createBuiltInNode<Variable>(StorageQualifier::In, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_InstanceID");
        vs_p[TGE_EFFECT_BUILTIN_OUT_GL_POSITION] = createBuiltInNode<Variable>(StorageQualifier::Out, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC4), "gl_Position");
        vs_p[TGE_EFFECT_BUILTIN_OUT_GL_POINT_SIZE] = createBuiltInNode<Variable>(StorageQualifier::Out, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_FLOAT), "gl_PointSize");
        auto arr_size = createBuiltInType<ArrayType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_FLOAT),
                                                                 AST::CreateNode<AST::Reference<Variable>>(TGE_DEFAULT_LOCATION, extractNode<Variable>(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_GL_MAX_CLIP_DISTANCE)));
        vs_p[TGE_EFFECT_BUILTIN_OUT_GL_CLIP_DISTANCE] = createBuiltInNode<Variable>(StorageQualifier::Out,
                                                                                   m_ObjectPool[arr_size].extract<Type>(),
                                                                                   "gl_ClipDistance");
    //  TGE_ASSERT(std::find(m_Stack.begin(), m_Stack.end(), StackASTNodePtr()) == m_Stack.end(), "Unfilled stack");
    } break;
    case ShaderType::FragmentShader:
    {
        size_t current_size = m_Stack.size();
        m_Stack.resize(m_Stack.size() + TGE_EFFECT_BUILTINS_FS);
        StackType::iterator::pointer fs_p = &m_Stack[current_size];
        fs_p[TGE_EFFECT_BUILTIN_DFDX] = m_FSBuiltIns[TGE_EFFECT_BUILTIN_DFDX];
        fs_p[TGE_EFFECT_BUILTIN_DFDY] = m_FSBuiltIns[TGE_EFFECT_BUILTIN_DFDY];
        fs_p[TGE_EFFECT_BUILTIN_FWIDTH] = m_FSBuiltIns[TGE_EFFECT_BUILTIN_FWIDTH];
        fs_p[TGE_EFFECT_BUILTIN_INTERPOLATEATCENTROID] = m_FSBuiltIns[TGE_EFFECT_BUILTIN_INTERPOLATEATCENTROID];
        fs_p[TGE_EFFECT_BUILTIN_INTERPOLATEATSAMPLE] = m_FSBuiltIns[TGE_EFFECT_BUILTIN_INTERPOLATEATSAMPLE];
        fs_p[TGE_EFFECT_BUILTIN_INTERPOLATEATOFFSET] = m_FSBuiltIns[TGE_EFFECT_BUILTIN_INTERPOLATEATOFFSET];
        fs_p[TGE_EFFECT_BUILTIN_IN_GL_FRAG_COORD] = createBuiltInNode<Variable>(StorageQualifier::In, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC4), "gl_FragCoord");
        fs_p[TGE_EFFECT_BUILTIN_IN_GL_FRONT_FACING] = createBuiltInNode<Variable>(StorageQualifier::In, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_BOOL), "gl_FrontFacing");
        auto cdist_arr_size = createBuiltInType<ArrayType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_FLOAT),
                                                                       AST::CreateNode<AST::Reference<Variable>>(TGE_DEFAULT_LOCATION, extractNode<Variable>(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_GL_MAX_CLIP_DISTANCE)));
        fs_p[TGE_EFFECT_BUILTIN_IN_GL_CLIP_DISTANCE] = createBuiltInNode<Variable>(StorageQualifier::In,
                                                                                m_ObjectPool[cdist_arr_size].extract<Type>(),
                                                                                "gl_ClipDistance");
        fs_p[TGE_EFFECT_BUILTIN_IN_GL_POINT_COORD] = createBuiltInNode<Variable>(StorageQualifier::In, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC4), "gl_PointCoord");
        fs_p[TGE_EFFECT_BUILTIN_IN_GL_PRIMITIVE_ID] = createBuiltInNode<Variable>(StorageQualifier::In, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_PrimitiveID");
        fs_p[TGE_EFFECT_BUILTIN_IN_GL_SAMPLE_ID] = createBuiltInNode<Variable>(StorageQualifier::In, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_INT), "gl_SampleID");
        fs_p[TGE_EFFECT_BUILTIN_IN_GL_SAMPLE_POSITION] = createBuiltInNode<Variable>(StorageQualifier::In, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_VEC2), "gl_SamplePosition");
        fs_p[TGE_EFFECT_BUILTIN_OUT_GL_FRAG_DEPTH] = createBuiltInNode<Variable>(StorageQualifier::Out, extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_FLOAT), "gl_FragDepth");
        auto samp_arr_size = createBuiltInType<ArrayType>(extractType(m_ShaderBuiltIns, TGE_EFFECT_BUILTIN_FLOAT),
                                                                      AST::Node()); // TODO: check whether it is working
        fs_p[TGE_EFFECT_BUILTIN_OUT_GL_SAMPLER_MASK] = createBuiltInNode<Variable>(StorageQualifier::Out,
                                                                                m_ObjectPool[samp_arr_size].extract<Type>(),
                                                                                "gl_SampleMask");
    //  TGE_ASSERT(std::find(m_Stack.begin(), m_Stack.end(), StackASTNodePtr()) == m_Stack.end(), "Unfilled stack");
    } break;
    default:
        TGE_ASSERT(false, "Unsupported shader"); break;
    }
}

void Driver::endShader()
{
    endBlock();
    TGE_ASSERT(m_StackPointers.empty(), "Expected empty shader stack");
}

void Driver::beginBlock()
{
    m_StackPointers.push_back(m_Stack.size());
}

void Driver::endBlock()
{
    m_Stack.erase(m_Stack.begin() + m_StackPointers.back(), m_Stack.end());
    m_StackPointers.pop_back();
}
}
}
