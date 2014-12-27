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

#if !defined(GL_LIBRARY_HH) || defined(TEMPEST_EXTRACT_FUNCTIONS)

#ifndef TEMPEST_EXTRACT_FUNCTIONS
#define GL_LIBRARY_HH
#include "tempest/utils/library.hh"
#include "tempest/utils/macros.hh"

#ifdef _WIN32
    #include <windows.h>
#endif

#include <stddef.h>

#ifndef APIENTRY
#   define APIENTRY
#endif

#ifndef DECLARE_GL_FUNCTION
#   define DECLARE_GL_FUNCTION(return_type, name, ...) typedef return_type (APIENTRY* PFN##name##PROC)(__VA_ARGS__); \
                                                       extern PFN##name##PROC name;
#   define DECLARE_GL_FUNCTION_OPTIONAL(caps, return_type, name, ...) typedef return_type (APIENTRY* PFN##name##PROC)(__VA_ARGS__); \
                                                                      extern PFN##name##PROC name;
#   define DECLARE_SYS_FUNCTION(return_type, name, ...) typedef return_type (APIENTRY* PFN##name##PROC)(__VA_ARGS__); \
                                                        extern PFN##name##PROC name;
#   define DECLARE_SYS_GL_FUNCTION(return_type, name, ...) typedef return_type (APIENTRY* PFN##name##PROC)(__VA_ARGS__); \
                                                           extern PFN##name##PROC name;
#endif

#ifdef _WIN32
#else
#include <X11/Xlib.h>
#endif

#include <sstream>
#include <iomanip>

#include "tempest/utils/assert.hh"
#include "tempest/utils/logging.hh"

namespace Tempest
{
class RenderingLibrary;
class WindowInfo;

class GLLibrary
{
#ifdef _WIN32
    HGLRC   m_RC = nullptr;
#elif !defined(LINUX)
#   error "Unsupported platform"
#endif
    Library m_GLLib;
public:
    explicit GLLibrary()=default;
     ~GLLibrary();

    bool initDeviceContextLibrary();
    bool initGraphicsLibrary();
};

typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef signed char GLbyte;
typedef char GLchar;
typedef short GLshort;
typedef int GLint;
typedef int GLsizei;
typedef unsigned char GLubyte;
typedef unsigned short GLushort;
typedef unsigned int GLuint;
typedef float GLfloat;
typedef float GLclampf;
typedef void GLvoid;
typedef int GLintptrARB;
typedef int GLsizeiptrARB;
typedef int GLfixed;
typedef int GLclampx;
typedef unsigned long long GLuint64;
typedef double GLdouble;
typedef double GLclampd;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;
typedef struct __GLsync *GLsync;

// _ARB or _EXT is appended to all names because otherwise we get name clashes
// because of bad system macros.

// You may use the standard references for more information about each macro. Just
// remove GL_ (actually the documentation omits it in most places) and append
// the relevant suffix.

// The C++11 enum classes give you better debug information and type safety.

enum class GLBlendFactorMode: GLuint
{
    GL_ZERO                            = 0,
    GL_ONE                             = 1,
    GL_SRC_COLOR                       = 0x0300,
    GL_ONE_MINUS_SRC_COLOR             = 0x0301,
    GL_DST_COLOR                       = 0x0306,
    GL_ONE_MINUS_DST_COLOR             = 0x0307,
    GL_SRC_ALPHA                       = 0x0302,
    GL_ONE_MINUS_SRC_ALPHA             = 0x0303,
    GL_DST_ALPHA                       = 0x0304,
    GL_ONE_MINUS_DST_ALPHA             = 0x0305,
    GL_CONSTANT_COLOR                  = 0x8001,
    GL_ONE_MINUS_CONSTANT_COLOR        = 0x8002,
    GL_CONSTANT_ALPHA                  = 0x8003,
    GL_ONE_MINUS_CONSTANT_ALPHA        = 0x8004,
    GL_SRC_ALPHA_SATURATE              = 0x0308,
    GL_SRC1_COLOR                      = 0x88F9,
    GL_ONE_MINUS_SRC1_COLOR            = 0x88FA,
    GL_ONE_MINUS_SRC1_ALPHA            = 0x88FB,
    GL_SRC1_ALPHA                      = 0x8589
};

enum class GLLogicOpMode: GLuint
{
    GL_CLEAR                           = 0x1500,
    GL_SET                             = 0x150F,
    GL_COPY                            = 0x1503,
    GL_COPY_INVERTED                   = 0x150C,
    GL_NOOP                            = 0x1505,
    GL_INVERT                          = 0x150A,
    GL_AND                             = 0x1501,
    GL_NAND                            = 0x150E,
    GL_OR                              = 0x1507,
    GL_NOR                             = 0x1508,
    GL_XOR                             = 0x1506,
    GL_EQUIV                           = 0x1509,
    GL_AND_REVERSE                     = 0x1502,
    GL_AND_INVERTED                    = 0x1504,
    GL_OR_REVERSE                      = 0x150B,
    GL_OR_INVERTED                     = 0x150D
};

enum class GLFaceMode: GLuint
{
    GL_FRONT_AND_BACK                  = 0x0408,
    GL_FRONT                           = 0x0404,
    GL_BACK                            = 0x0405
};

enum class GLBufferMode: GLuint
{
    GL_NONE                            = 0,
    GL_FRONT_LEFT                      = 0x0400,
    GL_FRONT_RIGHT                     = 0x0401,
    GL_BACK_LEFT                       = 0x0402,
    GL_BACK_RIGHT                      = 0x0403,
    GL_FRONT                           = 0x0404,
    GL_BACK                            = 0x0405,
    GL_LEFT                            = 0x0406,
    GL_RIGHT                           = 0x0407,
    GL_FRONT_AND_BACK                  = 0x0408
};

enum class GLOrderMode: GLuint
{
    GL_CW                              = 0x0900,
    GL_CCW                             = 0x0901
};

enum class GLCapabilityMode: GLuint
{
    GL_BLEND                           = 0x0BE2,
    GL_CLIP_DISTANCE0                  = 0x3000,
    GL_CLIP_DISTANCE1                  = 0x3001,
    GL_CLIP_DISTANCE2                  = 0x3002,
    GL_CLIP_DISTANCE3                  = 0x3003,
    GL_CLIP_DISTANCE4                  = 0x3004,
    GL_CLIP_DISTANCE5                  = 0x3005,
    GL_CLIP_DISTANCE6                  = 0x3006,
    GL_CLIP_DISTANCE7                  = 0x3007,
    GL_COLOR_LOGIC_OP                  = 0x0BF2,
    GL_CULL_FACE                       = 0x0B44,
    GL_DEBUG_OUTPUT                    = 0x92E0,
    GL_DEBUG_OUTPUT_SYNCHRONOUS        = 0x8242,
    GL_DEPTH_CLAMP                     = 0x864F,
    GL_DEPTH_TEST                      = 0x0B71,
    GL_DITHER                          = 0x0BD0,
    GL_FRAMEBUFFER_SRGB                = 0x8DB9,
    GL_LINE_SMOOTH                     = 0x0B20,
    GL_MULTISAMPLE                     = 0x809D,
    GL_POLYGON_OFFSET_FILL             = 0x8037,
    GL_POLYGON_OFFSET_LINE             = 0x2A02,
    GL_POLYGON_OFFSET_POINT            = 0x2A01,
    GL_POLYGON_SMOOTH                  = 0x0B41,
    GL_PRIMITIVE_RESTART               = 0x8F9D,
    GL_PRIMITIVE_RESTART_FIXED_INDEX   = 0x8D69,
    GL_RASTERIZER_DISCARD              = 0x8C89,
    GL_SAMPLE_ALPHA_TO_COVERAGE        = 0x809E,
    GL_SAMPLE_ALPHA_TO_ONE             = 0x809F,
    GL_SAMPLE_COVERAGE                 = 0x80A0,
    GL_SAMPLE_SHADING                  = 0x8C36,
    GL_SAMPLE_MASK                     = 0x8E51,
    GL_SCISSOR_TEST                    = 0x0C11,
    GL_STENCIL_TEST                    = 0x0B90,
    GL_TEXTURE_CUBE_MAP_SEAMLESS       = 0x884F,
    GL_PROGRAM_POINT_SIZE              = 0x8642
};

#define UINT_TO_GL_CLIP_DISTANCE(num) static_cast<GLCapabilityMode>(static_cast<GLuint>(GL_CLIP_DISTANCE0) + num)

enum class GLFillMode: GLuint
{
    GL_POINT                           = 0x1B00,
    GL_LINE                            = 0x1B01,
    GL_FILL                            = 0x1B02
};

enum class GLErrorCode: GLuint
{
    GL_NO_ERROR                        = 0,
    GL_INVALID_ENUM                    = 0x0500,
    GL_INVALID_VALUE                   = 0x0501,
    GL_INVALID_OPERATION               = 0x0502,
    GL_INVALID_FRAMEBUFFER_OPERATION   = 0x0506,
    GL_OUT_OF_MEMORY                   = 0x0505,
    GL_STACK_UNDERFLOW                 = 0x0504,
    GL_STACK_OVERFLOW                  = 0x0503
};

enum class GLHintTarget: GLuint
{
    GL_LINE_SMOOTH_HINT                = 0x0C52,
    GL_POLYGON_SMOOTH_HINT             = 0x0C53,
    GL_TEXTURE_COMPRESSION_HINT        = 0x84EF,
    GL_FRAGMENT_SHADER_DERIVATIVE_HINT = 0x8B8B
};

enum class GLHintMode: GLuint
{
    GL_FASTEST                         = 0x1101,
    GL_NICEST                          = 0x1102,
    GL_DONT_CARE                       = 0x1100
};

enum class GLComparisonFunction: GLuint
{
    GL_NEVER                           = 0x0200,
    GL_LESS                            = 0x0201,
    GL_EQUAL                           = 0x0202,
    GL_LEQUAL                          = 0x0203,
    GL_GREATER                         = 0x0204,
    GL_NOTEQUAL                        = 0x0205,
    GL_GEQUAL                          = 0x0206,
    GL_ALWAYS                          = 0x0207
};

enum class GLPixelStoreMode: GLuint
{
    GL_PACK_SWAP_BYTES                 = 0x0D00,
    GL_PACK_LSB_FIRST                  = 0x0D01,
    GL_PACK_ROW_LENGTH                 = 0x0D02,
    GL_PACK_SKIP_ROWS                  = 0x0D03,
    GL_PACK_SKIP_PIXELS                = 0x0D04,
    GL_PACK_SKIP_IMAGES                = 0x806B,
    GL_PACK_ALIGNMENT                  = 0x0D05,
    GL_PACK_IMAGE_HEIGHT               = 0x806C,
    GL_UNPACK_SWAP_BYTES               = 0x0CF0,
    GL_UNPACK_LSB_FIRST                = 0x0CF1,
    GL_UNPACK_ROW_LENGTH               = 0x0CF2,
    GL_UNPACK_SKIP_ROWS                = 0x0CF3,
    GL_UNPACK_SKIP_PIXELS              = 0x0CF4,
    GL_UNPACK_SKIP_IMAGES              = 0x806D,
    GL_UNPACK_ALIGNMENT                = 0x0CF5,
    GL_UNPACK_IMAGE_HEIGHT             = 0x806E
};

enum class GLStencilOpMode: GLuint
{
    GL_ZERO                            = 0,
    GL_KEEP                            = 0x1E00,
    GL_REPLACE                         = 0x1E01,
    GL_INCR                            = 0x1E02,
    GL_DECR                            = 0x1E03,
    GL_INCR_WRAP                       = 0x8507,
    GL_DECR_WRAP                       = 0x8508,
    GL_INVERT                          = 0x150A
};

enum class GLTextureTarget: GLuint
{
    GL_TEXTURE_1D                      = 0x0DE0,
    GL_TEXTURE_1D_ARRAY                = 0x8C18,
    GL_TEXTURE_2D                      = 0x0DE1,
    GL_TEXTURE_2D_ARRAY                = 0x8C1A,
    GL_TEXTURE_2D_MULTISAMPLE          = 0x9100,
    GL_TEXTURE_2D_MULTISAMPLE_ARRAY    = 0x9102,
    GL_TEXTURE_3D                      = 0x806F,
    GL_TEXTURE_CUBE_MAP                = 0x8513,
    GL_TEXTURE_CUBE_MAP_ARRAY          = 0x9009,
    GL_TEXTURE_RECTANGLE               = 0x84F5
};

enum class GLTextureParameter: GLuint
{
    GL_DEPTH_STENCIL_TEXTURE_MODE      = 0x90EA,
    GL_TEXTURE_BASE_LEVEL              = 0x813C,
    GL_TEXTURE_COMPARE_FUNC            = 0x884D,
    GL_TEXTURE_COMPARE_MODE            = 0x884C,
    GL_TEXTURE_LOD_BIAS                = 0x8501,
    GL_TEXTURE_MAG_FILTER              = 0x2800,
    GL_TEXTURE_MIN_FILTER              = 0x2801,
    GL_TEXTURE_MIN_LOD                 = 0x813A,
    GL_TEXTURE_MAX_LOD                 = 0x813B,
    GL_TEXTURE_MAX_LEVEL               = 0x813D,
    GL_TEXTURE_SWIZZLE_R               = 0x8E42,
    GL_TEXTURE_SWIZZLE_G               = 0x8E43,
    GL_TEXTURE_SWIZZLE_B               = 0x8E44,
    GL_TEXTURE_SWIZZLE_A               = 0x8E45,
    GL_TEXTURE_WRAP_S                  = 0x2802,
    GL_TEXTURE_WRAP_T                  = 0x2803,
    GL_TEXTURE_WRAP_R                  = 0x8072,
    GL_TEXTURE_MAX_ANISOTROPY_EXT      = 0x84FE,
    GL_TEXTURE_BORDER_COLOR            = 0x1004
};

enum class GLDrawMode: GLuint
{
    GL_POINTS                          = 0x0000,
    GL_LINES                           = 0x0001,
    GL_LINE_LOOP                       = 0x0002,
    GL_LINE_STRIP                      = 0x0003,
    GL_TRIANGLES                       = 0x0004,
    GL_TRIANGLE_STRIP                  = 0x0005,
    GL_TRIANGLE_FAN                    = 0x0006,
    GL_QUADS                           = 0x0007,
    GL_LINES_ADJACENCY                 = 0x000A,
    GL_LINE_STRIP_ADJACENCY            = 0x000B,
    GL_TRIANGLES_ADJACENCY             = 0x000C,
    GL_TRIANGLE_STRIP_ADJACENCY        = 0x000D,
    GL_PATCHES                         = 0x000E
};

enum class GLBlendEquationMode: GLuint
{
    GL_FUNC_ADD                        = 0x8006,
    GL_FUNC_SUBTRACT                   = 0x800A,
    GL_FUNC_REVERSE_SUBTRACT           = 0x800B,
    GL_MIN                             = 0x8007,
    GL_MAX                             = 0x8008
};

enum class GLTextureIndex: GLuint
{
    GL_TEXTURE0                        = 0x84C0,
    GL_TEXTURE1                        = 0x84C1,
    GL_TEXTURE2                        = 0x84C2,
    GL_TEXTURE3                        = 0x84C3,
    GL_TEXTURE4                        = 0x84C4,
    GL_TEXTURE5                        = 0x84C5,
    GL_TEXTURE6                        = 0x84C6,
    GL_TEXTURE7                        = 0x84C7,
    GL_TEXTURE8                        = 0x84C8,
    GL_TEXTURE9                        = 0x84C9,
    GL_TEXTURE10                       = 0x84CA,
    GL_TEXTURE11                       = 0x84CB,
    GL_TEXTURE12                       = 0x84CC,
    GL_TEXTURE13                       = 0x84CD,
    GL_TEXTURE14                       = 0x84CE,
    GL_TEXTURE15                       = 0x84CF,
    GL_TEXTURE16                       = 0x84D0,
    GL_TEXTURE17                       = 0x84D1,
    GL_TEXTURE18                       = 0x84D2,
    GL_TEXTURE19                       = 0x84D3,
    GL_TEXTURE20                       = 0x84D4,
    GL_TEXTURE21                       = 0x84D5,
    GL_TEXTURE22                       = 0x84D6,
    GL_TEXTURE23                       = 0x84D7,
    GL_TEXTURE24                       = 0x84D8,
    GL_TEXTURE25                       = 0x84D9,
    GL_TEXTURE26                       = 0x84DA,
    GL_TEXTURE27                       = 0x84DB,
    GL_TEXTURE28                       = 0x84DC,
    GL_TEXTURE29                       = 0x84DD,
    GL_TEXTURE30                       = 0x84DE,
    GL_TEXTURE31                       = 0x84DF
};

#define UINT_TO_GL_TEXTURE(num) static_cast<GLTextureIndex>(static_cast<GLuint>(GL_TEXTURE0) + num)

enum class GLType
{
    GL_BYTE                            = 0x1400,
    GL_UNSIGNED_BYTE                   = 0x1401,
    GL_SHORT                           = 0x1402,
    GL_UNSIGNED_SHORT                  = 0x1403,
    GL_INT                             = 0x1404,
    GL_UNSIGNED_INT                    = 0x1405,
    GL_FLOAT                           = 0x1406,
    GL_DOUBLE                          = 0x140A,
    GL_HALF_FLOAT                      = 0x140B,
    GL_UNSIGNED_BYTE_3_3_2             = 0x8032,
    GL_UNSIGNED_SHORT_4_4_4_4          = 0x8033,
    GL_UNSIGNED_SHORT_5_5_5_1          = 0x8034,
    GL_UNSIGNED_INT_8_8_8_8            = 0x8035,
    GL_UNSIGNED_INT_10_10_10_2         = 0x8036,
    GL_UNSIGNED_BYTE_2_3_3_REV         = 0x8362,
    GL_UNSIGNED_SHORT_5_6_5            = 0x8363,
    GL_UNSIGNED_SHORT_5_6_5_REV        = 0x8364,
    GL_UNSIGNED_SHORT_4_4_4_4_REV      = 0x8365,
    GL_UNSIGNED_SHORT_1_5_5_5_REV      = 0x8366,
    GL_UNSIGNED_INT_8_8_8_8_REV        = 0x8367,
    GL_UNSIGNED_INT_2_10_10_10_REV     = 0x8368,
    GL_UNSIGNED_INT_24_8               = 0x84FA,
    GL_UNSIGNED_INT_10F_11F_11F_REV    = 0x8C3B,
    GL_UNSIGNED_INT_5_9_9_9_REV        = 0x8C3E,
    GL_FLOAT_32_UNSIGNED_INT_24_8_REV  = 0x8DAD,
    GL_DEPTH_COMPONENT16               = 0x81A5,
    GL_DEPTH_COMPONENT24               = 0x81A6,
    GL_DEPTH_COMPONENT32               = 0x81A7,
    GL_INT_2_10_10_10_REV              = 0x8D9F
};

enum class GLQueryTarget: GLuint
{
    GL_SAMPLES_PASSED                  = 0x8914,
    GL_ANY_SAMPLES_PASSED              = 0x8C2F,
    GL_ANY_SAMPLES_PASSED_CONSERVATIVE = 0x8D6A,
    GL_PRIMITIVES_GENERATED            = 0x8C87,
    GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 0x8C88,
    GL_TIME_ELAPSED                    = 0x88BF
};

enum class GLQueryParameter: GLuint
{
    GL_QUERY_RESULT                    = 0x8866,
    GL_QUERY_RESULT_NO_WAIT            = 0x9194,
    GL_QUERY_RESULT_AVAILABLE          = 0x8867
};

enum class GLBufferTarget: GLuint
{
    GL_ARRAY_BUFFER                    = 0x8892,
    GL_ATOMIC_COUNTER_BUFFER           = 0x92C0,
    GL_COPY_READ_BUFFER                = 0x8F36,
    GL_COPY_WRITE_BUFFER               = 0x8F37,
    GL_DISPATCH_INDIRECT_BUFFER        = 0x90EE,
    GL_DRAW_INDIRECT_BUFFER            = 0x8F3F,
    GL_ELEMENT_ARRAY_BUFFER            = 0x8893,
    GL_PIXEL_PACK_BUFFER               = 0x88EB,
    GL_PIXEL_UNPACK_BUFFER             = 0x88EC,
    GL_QUERY_BUFFER                    = 0x9192,
    GL_SHADER_STORAGE_BUFFER           = 0x90D2,
    GL_TEXTURE_BUFFER                  = 0x8C2A,
    GL_TRANSFORM_FEEDBACK_BUFFER       = 0x8C8E,
    GL_UNIFORM_BUFFER                  = 0x8A11
};

enum class GLAccessMode: GLuint
{
    GL_READ_ONLY                       = 0x88B8,
    GL_WRITE_ONLY                      = 0x88B9,
    GL_READ_WRITE                      = 0x88BA
};

enum class GLFormat: GLuint
{
    GL_STENCIL_INDEX                   = 0x1901,
    GL_DEPTH_COMPONENT                 = 0x1902,
    GL_DEPTH_STENCIL                   = 0x84F9,
    GL_RED                             = 0x1903,
    GL_GREEN                           = 0x1904,
    GL_BLUE                            = 0x1905,
    GL_ALPHA                           = 0x1906,
    GL_RGB                             = 0x1907,
    GL_RGBA                            = 0x1908,
    GL_BGR                             = 0x80E0,
    GL_BGRA                            = 0x80E1,
    GL_RG                              = 0x8227,
    GL_RG_INTEGER                      = 0x8228,
    GL_R8                              = 0x8229,
    GL_R16                             = 0x822A,
    GL_RG8                             = 0x822B,
    GL_RG16                            = 0x822C,
    GL_R16F                            = 0x822D,
    GL_R32F                            = 0x822E,
    GL_RG16F                           = 0x822F,
    GL_RG32F                           = 0x8230,
    GL_R8I                             = 0x8231,
    GL_R8UI                            = 0x8232,
    GL_R16I                            = 0x8233,
    GL_R16UI                           = 0x8234,
    GL_R32I                            = 0x8235,
    GL_R32UI                           = 0x8236,
    GL_RG8I                            = 0x8237,
    GL_RG8UI                           = 0x8238,
    GL_RG16I                           = 0x8239,
    GL_RG16UI                          = 0x823A,
    GL_RG32I                           = 0x823B,
    GL_RG32UI                          = 0x823C,
    GL_R11F_G11F_B10F                  = 0x8C3A,
    GL_RGB9_E5                         = 0x8C3D,
    GL_RGBA32F                         = 0x8814,
    GL_RGB32F                          = 0x8815,
    GL_RGBA16F                         = 0x881A,
    GL_RGB16F                          = 0x881B,
    GL_R8_SNORM                        = 0x8F94,
    GL_RG8_SNORM                       = 0x8F95,
    GL_RGB8_SNORM                      = 0x8F96,
    GL_RGBA8_SNORM                     = 0x8F97,
    GL_R16_SNORM                       = 0x8F98,
    GL_RG16_SNORM                      = 0x8F99,
    GL_RGB16_SNORM                     = 0x8F9A,
    GL_RGBA16_SNORM                    = 0x8F9B,
    GL_R3_G3_B2                        = 0x2A10,
    GL_RGB4                            = 0x804F,
    GL_RGB5                            = 0x8050,
    GL_RGB8                            = 0x8051,
    GL_RGB10                           = 0x8052,
    GL_RGB12                           = 0x8053,
    GL_RGB16                           = 0x8054,
    GL_RGBA2                           = 0x8055,
    GL_RGBA4                           = 0x8056,
    GL_RGB5_A1                         = 0x8057,
    GL_RGBA8                           = 0x8058,
    GL_RGB10_A2                        = 0x8059,
    GL_RGBA12                          = 0x805A,
    GL_RGBA16                          = 0x805B,
    GL_SRGB                            = 0x8C40,
    GL_SRGB8                           = 0x8C41,
    GL_SRGB_ALPHA                      = 0x8C42,
    GL_SRGB8_ALPHA8                    = 0x8C43,
    GL_COMPRESSED_RED                  = 0x8225,
    GL_COMPRESSED_RG                   = 0x8226,
    GL_COMPRESSED_RGB                  = 0x84ED,
    GL_COMPRESSED_RGBA                 = 0x84EE,
    GL_COMPRESSED_SRGB                 = 0x8C48,
    GL_COMPRESSED_SRGB_ALPHA           = 0x8C49,
    GL_COMPRESSED_RED_RGTC1            = 0x8DBB,
    GL_COMPRESSED_SIGNED_RED_RGTC1     = 0x8DBC,
    GL_COMPRESSED_RG_RGTC2             = 0x8DBD,
    GL_COMPRESSED_SIGNED_RG_RGTC2      = 0x8DBE,
    GL_COMPRESSED_RGBA_BPTC_UNORM      = 0x8E8C,
    GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM = 0x8E8D,
    GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT = 0x8E8E,
    GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT = 0x8E8F,
    GL_DEPTH_COMPONENT16               = 0x81A5,
    GL_DEPTH_COMPONENT24               = 0x81A6,
    GL_DEPTH_COMPONENT32               = 0x81A7,
    GL_DEPTH24_STENCIL8                = 0x88F0
};

enum class GLUsageMode: GLuint
{
    GL_STREAM_DRAW                     = 0x88E0,
    GL_STREAM_READ                     = 0x88E1,
    GL_STREAM_COPY                     = 0x88E2,
    GL_STATIC_DRAW                     = 0x88E4,
    GL_STATIC_READ                     = 0x88E5,
    GL_STATIC_COPY                     = 0x88E6,
    GL_DYNAMIC_DRAW                    = 0x88E8,
    GL_DYNAMIC_READ                    = 0x88E9,
    GL_DYNAMIC_COPY                    = 0x88EA
};

enum class GLDebugSourceType: GLuint
{
    GL_DEBUG_SOURCE_API                = 0x8246,
    GL_DEBUG_SOURCE_WINDOW_SYSTEM      = 0x8247,
    GL_DEBUG_SOURCE_SHADER_COMPILER    = 0x8248,
    GL_DEBUG_SOURCE_THIRD_PARTY        = 0x8249,
    GL_DEBUG_SOURCE_APPLICATION        = 0x824A,
    GL_DEBUG_SOURCE_OTHER              = 0x824B
};

enum class GLDebugType: GLuint
{
    GL_DEBUG_TYPE_ERROR                = 0x824C,
    GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR  = 0x824D,
    GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR   = 0x824E,
    GL_DEBUG_TYPE_PORTABILITY          = 0x824F,
    GL_DEBUG_TYPE_PERFORMANCE          = 0x8250,
    GL_DEBUG_TYPE_MARKER               = 0x8268,
    GL_DEBUG_TYPE_PUSH_GROUP           = 0x8269,
    GL_DEBUG_TYPE_POP_GROUP            = 0x826A,
    GL_DEBUG_TYPE_OTHER                = 0x8251
};

enum class GLSeverityType: GLuint
{
    GL_DEBUG_SEVERITY_HIGH             = 0x9146,
    GL_DEBUG_SEVERITY_MEDIUM           = 0x9147,
    GL_DEBUG_SEVERITY_LOW              = 0x9148,
    GL_DEBUG_SEVERITY_NOTIFICATION     = 0x826B
};

enum GLContextAttribute
{
    WGL_CONTEXT_MAJOR_VERSION_ARB      = 0x2091,
    WGL_CONTEXT_MINOR_VERSION_ARB      = 0x2092,
    WGL_CONTEXT_LAYER_PLANE_ARB        = 0x2093,
    WGL_CONTEXT_FLAGS_ARB              = 0x2094,
    WGL_CONTEXT_PROFILE_MASK_ARB       = 0x9126,
    WGL_CONTEXT_CORE_PROFILE_BIT_ARB          = 0x00000001,
    WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB = 0x00000002,
    WGL_CONTEXT_DEBUG_BIT_ARB                 = 0x00000001,
    WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB    = 0x00000002
};

enum class GLClientState: GLuint
{
    GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV = 0x8F1E,
    GL_ELEMENT_ARRAY_UNIFIED_NV       = 0x8F1F
};

enum class GLBufferContentType: GLuint
{
    GL_COLOR                                  = 0x1800,
    GL_DEPTH                                  = 0x1801,
    GL_STENCIL                                = 0x1802,
    GL_DEPTH_STENCIL                          = 0x84F9
};

enum class GLBufferParameterNV: GLuint
{
    GL_BUFFER_GPU_ADDRESS_NV                  = 0x8F1D
};

enum class GLSyncCondition: GLuint
{
    GL_SYNC_GPU_COMMANDS_COMPLETE             = 0x9117
};

enum class GLWaitStatus: GLuint
{
    GL_ALREADY_SIGNALED                       = 0x911A,
    GL_TIMEOUT_EXPIRED                        = 0x911B,
    GL_CONDITION_SATISFIED                    = 0x911C,
    GL_WAIT_FAILED                            = 0x911D
};

enum GLSyncBitfield
{
    GL_SYNC_FLUSH_COMMANDS_BIT                = 0x00000001
};

enum class GLShaderType: GLuint
{
    GL_VERTEX_SHADER                          = 0x8B31,
    GL_TESS_CONTROL_SHADER                    = 0x8E88,
    GL_TESS_EVALUATION_SHADER                 = 0x8E87,
    GL_GEOMETRY_SHADER                        = 0x8DD9,
    GL_FRAGMENT_SHADER                        = 0x8B30,
    GL_COMPUTE_SHADER                         = 0x91B9
};

enum class GLShaderParameter: GLuint
{
    GL_SHADER_TYPE                            = 0x8B4F,
    GL_DELETE_STATUS                          = 0x8B80,
    GL_COMPILE_STATUS                         = 0x8B81,
    GL_INFO_LOG_LENGTH                        = 0x8B84,
    GL_SHADER_SOURCE_LENGTH                   = 0x8B88
};

enum
{
    GL_FALSE                                  = 0,
    GL_TRUE                                   = 1
};

enum class GLProgramParameter: GLuint
{
    GL_DELETE_STATUS                          = 0x8B80,
    GL_LINK_STATUS                            = 0x8B82,
    GL_VALIDATE_STATUS                        = 0x8B83,
    GL_INFO_LOG_LENGTH                        = 0x8B84,
    GL_ATTACHED_SHADERS                       = 0x8B85,
    GL_ACTIVE_ATTRIBUTES                      = 0x8B89,
    GL_ACTIVE_ATTRIBUTE_MAX_LENGTH            = 0x8B8A,
    GL_ACTIVE_UNIFORMS                        = 0x8B86,
    GL_ACTIVE_UNIFORM_MAX_LENGTH              = 0x8B87
};

enum
{
    WGL_NUMBER_PIXEL_FORMATS_ARB              = 0x2000,
    WGL_DRAW_TO_WINDOW_ARB                    = 0x2001,
    WGL_DRAW_TO_BITMAP_ARB                    = 0x2002,
    WGL_ACCELERATION_ARB                      = 0x2003,
    WGL_NEED_PALETTE_ARB                      = 0x2004,
    WGL_NEED_SYSTEM_PALETTE_ARB               = 0x2005,
    WGL_SWAP_LAYER_BUFFERS_ARB                = 0x2006,
    WGL_SWAP_METHOD_ARB                       = 0x2007,
    WGL_NUMBER_OVERLAYS_ARB                   = 0x2008,
    WGL_NUMBER_UNDERLAYS_ARB                  = 0x2009,
    WGL_TRANSPARENT_ARB                       = 0x200A,
    WGL_TRANSPARENT_RED_VALUE_ARB             = 0x2037,
    WGL_TRANSPARENT_GREEN_VALUE_ARB           = 0x2038,
    WGL_TRANSPARENT_BLUE_VALUE_ARB            = 0x2039,
    WGL_TRANSPARENT_ALPHA_VALUE_ARB           = 0x203A,
    WGL_TRANSPARENT_INDEX_VALUE_ARB           = 0x203B,
    WGL_SHARE_DEPTH_ARB                       = 0x200C,
    WGL_SHARE_STENCIL_ARB                     = 0x200D,
    WGL_SHARE_ACCUM_ARB                       = 0x200E,
    WGL_SUPPORT_GDI_ARB                       = 0x200F,
    WGL_SUPPORT_OPENGL_ARB                    = 0x2010,
    WGL_DOUBLE_BUFFER_ARB                     = 0x2011,
    WGL_STEREO_ARB                            = 0x2012,
    WGL_PIXEL_TYPE_ARB                        = 0x2013,
    WGL_COLOR_BITS_ARB                        = 0x2014,
    WGL_RED_BITS_ARB                          = 0x2015,
    WGL_RED_SHIFT_ARB                         = 0x2016,
    WGL_GREEN_BITS_ARB                        = 0x2017,
    WGL_GREEN_SHIFT_ARB                       = 0x2018,
    WGL_BLUE_BITS_ARB                         = 0x2019,
    WGL_BLUE_SHIFT_ARB                        = 0x201A,
    WGL_ALPHA_BITS_ARB                        = 0x201B,
    WGL_ALPHA_SHIFT_ARB                       = 0x201C,
    WGL_ACCUM_BITS_ARB                        = 0x201D,
    WGL_ACCUM_RED_BITS_ARB                    = 0x201E,
    WGL_ACCUM_GREEN_BITS_ARB                  = 0x201F,
    WGL_ACCUM_BLUE_BITS_ARB                   = 0x2020,
    WGL_ACCUM_ALPHA_BITS_ARB                  = 0x2021,
    WGL_DEPTH_BITS_ARB                        = 0x2022,
    WGL_STENCIL_BITS_ARB                      = 0x2023,
    WGL_AUX_BUFFERS_ARB                       = 0x2024,
    WGL_NO_ACCELERATION_ARB                   = 0x2025,
    WGL_GENERIC_ACCELERATION_ARB              = 0x2026,
    WGL_FULL_ACCELERATION_ARB                 = 0x2027,
    WGL_SWAP_EXCHANGE_ARB                     = 0x2028,
    WGL_SWAP_COPY_ARB                         = 0x2029,
    WGL_SWAP_UNDEFINED_ARB                    = 0x202A,
    WGL_TYPE_RGBA_ARB                         = 0x202B,
    WGL_TYPE_COLORINDEX_ARB                   = 0x202C,
    WGL_SAMPLE_BUFFERS_ARB                    = 0x2041,
    WGL_SAMPLES_ARB                           = 0x2042
};

enum
{
    GL_NONE                                   = 0,
    GL_RED                                    = 0x1903,
    GL_GREEN                                  = 0x1904,
    GL_BLUE                                   = 0x1905,
    GL_ALPHA                                  = 0x1906,
    GL_NEAREST                                = 0x2600,
    GL_LINEAR                                 = 0x2601,
    GL_NEAREST_MIPMAP_NEAREST                 = 0x2700,
    GL_LINEAR_MIPMAP_NEAREST                  = 0x2701,
    GL_NEAREST_MIPMAP_LINEAR                  = 0x2702,
    GL_LINEAR_MIPMAP_LINEAR                   = 0x2703,
    GL_COMPARE_REF_TO_TEXTURE                 = 0x884E,
    GL_REPEAT                                 = 0x2901,
    GL_MIRRORED_REPEAT                        = 0x8370,
    GL_CLAMP_TO_EDGE                          = 0x812F,
    GL_CLAMP_TO_BORDER                        = 0x812D,
    GL_MIRROR_CLAMP_TO_EDGE                   = 0x8743
};

enum
{
    GL_DYNAMIC_STORAGE_BIT                    = 0x0100,
    GL_MAP_READ_BIT                           = 0x0001,
    GL_MAP_WRITE_BIT                          = 0x0002,
    GL_MAP_PERSISTENT_BIT                     = 0x0040,
    GL_MAP_COHERENT_BIT                       = 0x0080,
    GL_CLIENT_STORAGE_BIT                     = 0x0200
};

enum
{
    TEMPEST_GL_CAPS_330                       = 1 << 0,
    TEMPEST_GL_CAPS_400                       = 1 << 1,
    TEMPEST_GL_CAPS_410                       = 1 << 2,
    TEMPEST_GL_CAPS_420                       = 1 << 3,
    TEMPEST_GL_CAPS_430                       = 1 << 4,
    TEMPEST_GL_CAPS_440                       = 1 << 5,
    TEMPEST_GL_CAPS_450                       = 1 << 6,
    TEMPEST_GL_CAPS_TEXTURE_BINDLESS          = 1 << 10,
    TEMPEST_GL_CAPS_MDI_BINDLESS              = 1 << 11
};

bool IsGLCapabilitySupported(uint64 caps);

typedef void (APIENTRY  *GLDEBUGPROCARB)(GLDebugSourceType source, GLDebugType type, GLuint id, GLSeverityType severity, GLsizei length, const GLchar *message, const void *userParam);

// HACKS
BOOL w32hackChoosePixelFormat(HDC hDC, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
HGLRC w32hackCreateContextAttribs(HDC hDC, HGLRC hShareContext, const int *attribList);

typedef HGLRC(WINAPI* PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC hDC, HGLRC hShareContext, const int *attribList);
typedef BOOL(WINAPI* PFNWGLCHOOSEPIXELFORMATARBPROC)(HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
extern PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB;
extern PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB;
#endif

///////////////////
// WGL functions //
///////////////////
#ifdef _WIN32
DECLARE_SYS_FUNCTION(BOOL, wglMakeCurrent, HDC hdc, HGLRC hglrc);
DECLARE_SYS_FUNCTION(BOOL, wglDeleteContext, HGLRC hglrc);
DECLARE_SYS_FUNCTION(HGLRC, wglCreateContext, HDC hdc);
DECLARE_SYS_FUNCTION(PROC, wglGetProcAddress, LPCSTR lpszProc);
#endif

///////////////////
// GLX functions //
///////////////////
#ifndef _WIN32
DECLARE_SYS_FUNCTION(ProcType, glXGetProcAddress, const GLubyte* procName);
DECLARE_SYS_GL_FUNCTION(void, glXSwapBuffers, Display* dpy, GLXDrawable drawable);
DECLARE_SYS_GL_FUNCTION(void, glXDestroyContext, Display* dpy, GLXContext ctx);
DECLARE_SYS_GL_FUNCTION(Bool, glXMakeCurrent, Display* dpy, GLXDrawable drawable, GLXContext ctx);
DECLARE_SYS_GL_FUNCTION(const char*, glXQueryExtensionsString, Display* dpy, int screen);
DECLARE_SYS_GL_FUNCTION(GLXContext, glXCreateContextAttribsARB, Display *dpy, GLXFBConfig config, GLXContext share_context, Bool direct, const int *attrib_list);
DECLARE_SYS_GL_FUNCTION(Bool, glXQueryVersion, Display* dpy, int* major, int* minor);
DECLARE_SYS_GL_FUNCTION(GLXFBConfig*, glXChooseFBConfig, Display* dpy, int screen, const int* attrib_list, int* nelements);
DECLARE_SYS_GL_FUNCTION(int, PFNGLXGETFBCONFIGATTRIBPROC, Display* dpy,GLXFBConfig config, int attribute, int* value);
DECLARE_SYS_GL_FUNCTION(XVisualInfo*, glXGetVisualFromFBConfig, Display* dpy, GLXFBConfig config);
DECLARE_SYS_GL_FUNCTION(GLXContext, glXCreateNewContext, Display *dpy, GLXFBConfig config, int renderType, GLXContext shareList, Bool direct );
#endif

//////////////////////
// OpenGL functions //
//////////////////////
// Don't add Get functions without good reason!
DECLARE_GL_FUNCTION(void, glClearColor, GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
DECLARE_GL_FUNCTION(void, glClear, GLbitfield mask);
DECLARE_GL_FUNCTION(void, glColorMask, GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha );
DECLARE_GL_FUNCTION(void, glBlendFunc, GLBlendFactorMode sfactor, GLBlendFactorMode dfactor);
DECLARE_GL_FUNCTION(void, glLogicOp, GLLogicOpMode opcode);
DECLARE_GL_FUNCTION(void, glCullFace, GLFaceMode mode);
DECLARE_GL_FUNCTION(void, glFrontFace, GLOrderMode mode);
DECLARE_GL_FUNCTION(void, glLineWidth, GLfloat width);
DECLARE_GL_FUNCTION(void, glPolygonMode, GLFaceMode face, GLFillMode mode );
DECLARE_GL_FUNCTION(void, glPolygonOffset, GLfloat factor, GLfloat units);
DECLARE_GL_FUNCTION(void, glScissor, GLint x, GLint y, GLsizei width, GLsizei height);
DECLARE_GL_FUNCTION(void, glDrawBuffer, GLBufferMode mode);
DECLARE_GL_FUNCTION(void, glReadBuffer, GLBufferMode mode);
DECLARE_GL_FUNCTION(void, glEnable, GLCapabilityMode cap);
DECLARE_GL_FUNCTION(void, glDisable, GLCapabilityMode cap);
DECLARE_GL_FUNCTION(void, glEnablei, GLCapabilityMode target, GLuint index);
DECLARE_GL_FUNCTION(void, glDisablei, GLCapabilityMode target, GLuint index);
DECLARE_GL_FUNCTION(GLErrorCode, glGetError, void);
DECLARE_GL_FUNCTION(void, glFinish, void);
DECLARE_GL_FUNCTION(void, glFlush, void);
DECLARE_GL_FUNCTION(void, glHint, GLHintTarget target, GLHintMode mode );

DECLARE_GL_FUNCTION(void, glClearDepth, GLclampd depth);
DECLARE_GL_FUNCTION(void, glDepthFunc, GLComparisonFunction func);
DECLARE_GL_FUNCTION(void, glDepthMask, GLboolean flag);
DECLARE_GL_FUNCTION(void, glDepthRange, GLclampd near_val, GLclampd far_val);

DECLARE_GL_FUNCTION(void, glPixelStoref, GLPixelStoreMode pname, GLfloat param);
DECLARE_GL_FUNCTION(void, glPixelStorei, GLPixelStoreMode pname, GLint param);
DECLARE_GL_FUNCTION(void, glReadPixels, GLint x, GLint y, GLsizei width, GLsizei height, GLFormat format, GLType type, GLvoid *pixels);

DECLARE_GL_FUNCTION(void, glStencilFunc, GLComparisonFunction func, GLint ref, GLuint mask);
DECLARE_GL_FUNCTION(void, glStencilMask, GLuint mask );
DECLARE_GL_FUNCTION(void, glStencilOp, GLStencilOpMode fail, GLStencilOpMode zfail, GLStencilOpMode zpass);
DECLARE_GL_FUNCTION(void, glClearStencil, GLint s);

DECLARE_GL_FUNCTION(void, glTexParameterf, GLTextureTarget target, GLTextureParameter pname, GLfloat param);
DECLARE_GL_FUNCTION(void, glTexParameteri, GLTextureTarget target, GLTextureParameter pname, GLint param);
DECLARE_GL_FUNCTION(void, glTexParameterfv, GLTextureTarget target, GLTextureParameter pname, const GLfloat *params);
DECLARE_GL_FUNCTION(void, glTexParameteriv, GLTextureTarget target, GLTextureParameter pname, const GLint *params);
DECLARE_GL_FUNCTION(void, glTexImage1D, GLTextureTarget target, GLint level, GLFormat internalFormat, GLsizei width, GLint border, GLFormat format, GLType type, const GLvoid *pixels);
DECLARE_GL_FUNCTION(void, glTexImage2D, GLTextureTarget target, GLint level, GLFormat internalFormat, GLsizei width, GLsizei height, GLint border, GLFormat format, GLType type, const GLvoid *pixels);

DECLARE_GL_FUNCTION(void, glGenTextures, GLsizei n, GLuint *textures);
DECLARE_GL_FUNCTION(void, glDeleteTextures, GLsizei n, const GLuint *textures);
DECLARE_GL_FUNCTION(void, glBindTexture, GLTextureTarget target, GLuint texture);
DECLARE_GL_FUNCTION(GLboolean, glIsTexture, GLuint texture); // Use for debug only

DECLARE_GL_FUNCTION(void, glTexSubImage1D, GLTextureTarget target, GLint level, GLint xoffset, GLsizei width, GLFormat format, GLType type, const GLvoid *pixels);
DECLARE_GL_FUNCTION(void, glTexSubImage2D, GLTextureTarget target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLFormat format, GLType type, const GLvoid *pixels);
DECLARE_GL_FUNCTION(void, glCopyTexImage1D, GLTextureTarget target, GLint level, GLFormat internalformat, GLint x, GLint y, GLsizei width, GLint border);
DECLARE_GL_FUNCTION(void, glCopyTexImage2D, GLTextureTarget target, GLint level, GLFormat internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border);
DECLARE_GL_FUNCTION(void, glCopyTexSubImage1D, GLTextureTarget target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
DECLARE_GL_FUNCTION(void, glCopyTexSubImage2D, GLTextureTarget target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);

DECLARE_GL_FUNCTION(void, glViewport, GLint x, GLint y, GLsizei width, GLsizei height);

DECLARE_GL_FUNCTION(void, glTexImage3D, GLTextureTarget target, GLint level, GLFormat internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLFormat format, GLType type, const GLvoid *pixels );
DECLARE_GL_FUNCTION(void, glTexSubImage3D, GLTextureTarget target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLFormat format, GLType type, const GLvoid *pixels);
DECLARE_GL_FUNCTION(void, glCopyTexSubImage3D, GLTextureTarget target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);

DECLARE_GL_FUNCTION(void, glBlendEquation, GLBlendEquationMode mode);
DECLARE_GL_FUNCTION(void, glBlendColor, GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);

DECLARE_GL_FUNCTION(void, glActiveTexture, GLTextureIndex texture);
DECLARE_GL_FUNCTION(void, glCompressedTexImage1D, GLTextureTarget target, GLint level, GLFormat internalformat, GLsizei width, GLint border, GLsizei imageSize, const GLvoid *data);
DECLARE_GL_FUNCTION(void, glCompressedTexImage2D, GLTextureTarget target, GLint level, GLFormat internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const GLvoid *data);
DECLARE_GL_FUNCTION(void, glCompressedTexImage3D, GLTextureTarget target, GLint level, GLFormat internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid *data);
DECLARE_GL_FUNCTION(void, glCompressedTexSubImage1D, GLTextureTarget target, GLint level, GLint xoffset, GLsizei width, GLFormat format, GLsizei imageSize, const GLvoid *data);
DECLARE_GL_FUNCTION(void, glCompressedTexSubImage2D, GLTextureTarget target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLFormat format, GLsizei imageSize, const GLvoid *data);
DECLARE_GL_FUNCTION(void, glCompressedTexSubImage3D, GLTextureTarget target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLFormat format, GLsizei imageSize, const GLvoid *data);
DECLARE_GL_FUNCTION(void, glSampleCoverage, GLclampf value, GLboolean invert);

DECLARE_GL_FUNCTION(void, glDrawArrays, GLDrawMode mode, GLint first, GLsizei count);
DECLARE_GL_FUNCTION(void, glDrawRangeElements, GLDrawMode mode, GLuint start, GLuint end, GLsizei count, GLType type, const GLvoid *indices);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_400, void, glDrawElementsIndirect, GLDrawMode mode, GLType type, const void *indirect);
DECLARE_GL_FUNCTION(void, glDrawElementsBaseVertex, GLDrawMode mode, GLsizei count, GLType type, GLvoid *indices, GLint basevertex);

DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_400, void, glBlendFunci, GLuint buf, GLBlendFactorMode src, GLBlendFactorMode dst);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_400, void, glBlendEquationi, GLuint buf, GLBlendEquationMode mode);

DECLARE_GL_FUNCTION(void, glMultiDrawArrays, GLDrawMode mode, const GLint *first, const GLsizei *count, GLsizei drawcount);
DECLARE_GL_FUNCTION(void, glMultiDrawElements, GLDrawMode mode, const GLsizei *count, GLType type, const void *const*indices, GLsizei drawcount);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_430, void, glMultiDrawElementsIndirect, GLDrawMode mode, GLType type, const void *indirect, GLsizei drawcount, GLsizei stride);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_MDI_BINDLESS, void, glMultiDrawElementsIndirectBindlessNV, GLDrawMode mode, GLType type, const void *indirect, GLsizei drawCount, GLsizei stride, int vertexBufferCount);

DECLARE_GL_FUNCTION(void, glGenQueries, GLsizei n, GLuint *ids);
DECLARE_GL_FUNCTION(void, glDeleteQueries, GLsizei n, const GLuint *ids);
DECLARE_GL_FUNCTION(GLboolean, glIsQuery, GLuint id);
DECLARE_GL_FUNCTION(void, glBeginQuery, GLQueryTarget target, GLuint id);
DECLARE_GL_FUNCTION(void, glEndQuery, GLQueryTarget target);
DECLARE_GL_FUNCTION(void, glGetQueryObjectiv, GLuint id, GLQueryParameter pname, GLint *params);
DECLARE_GL_FUNCTION(void, glGetQueryObjectuiv, GLuint id, GLQueryParameter pname, GLuint *params);

DECLARE_GL_FUNCTION(void, glBindBuffer, GLBufferTarget target, GLuint buffer);
DECLARE_GL_FUNCTION(void, glDeleteBuffers, GLsizei n, const GLuint *buffers);
DECLARE_GL_FUNCTION(void, glGenBuffers, GLsizei n, GLuint *buffers);
DECLARE_GL_FUNCTION(GLboolean, glIsBuffer, GLuint buffer);
DECLARE_GL_FUNCTION(void, glBufferData, GLBufferTarget target, GLsizeiptr size, const void *data, GLUsageMode usage);
DECLARE_GL_FUNCTION(void, glBufferSubData, GLBufferTarget target, GLintptr offset, GLsizeiptr size, const void *data);
DECLARE_GL_FUNCTION(void *, glMapBuffer, GLBufferTarget target, GLAccessMode access);
DECLARE_GL_FUNCTION(GLboolean, glUnmapBuffer, GLBufferTarget target);

DECLARE_GL_FUNCTION(void, glStencilOpSeparate, GLFaceMode face, GLStencilOpMode sfail, GLStencilOpMode dpfail, GLStencilOpMode dppass);
DECLARE_GL_FUNCTION(void, glStencilFuncSeparate, GLFaceMode face, GLComparisonFunction func, GLint ref, GLuint mask);
DECLARE_GL_FUNCTION(void, glStencilMaskSeparate, GLFaceMode face, GLuint mask);

DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_430, void, glDebugMessageCallback, GLDEBUGPROCARB callback, const void *userParam);
DECLARE_GL_FUNCTION(void, glEnableClientState, GLClientState array);
DECLARE_GL_FUNCTION(void, glDisableClientState, GLClientState array);

DECLARE_GL_FUNCTION(void, glClearBufferfv, GLBufferContentType buffer, GLint drawbuffer, const GLfloat *value);
DECLARE_GL_FUNCTION(void, glClearBufferfi, GLBufferContentType buffer, GLint drawbuffer, GLfloat depth, GLint stencil);

DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_MDI_BINDLESS, void, glGetBufferParameterui64vNV, GLBufferTarget target, GLBufferParameterNV pname, GLuint64 *params);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_MDI_BINDLESS, void, glMakeBufferResidentNV, GLBufferTarget target, GLAccessMode access);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_430, void, glBindVertexBuffer, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride);

DECLARE_GL_FUNCTION(GLsync, glFenceSync, GLSyncCondition condition, GLbitfield flags);
DECLARE_GL_FUNCTION(GLboolean, glIsSync, GLsync sync);
DECLARE_GL_FUNCTION(void, glDeleteSync, GLsync sync);
DECLARE_GL_FUNCTION(GLWaitStatus, glClientWaitSync, GLsync sync, GLbitfield flags, GLuint64 timeout);
DECLARE_GL_FUNCTION(void, glBindBufferRange, GLBufferTarget target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);
DECLARE_GL_FUNCTION(GLuint, glCreateProgram, void);
DECLARE_GL_FUNCTION(void, glDeleteProgram, GLuint program);
DECLARE_GL_FUNCTION(GLuint, glCreateShader, GLShaderType type);
DECLARE_GL_FUNCTION(void, glShaderSource, GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
DECLARE_GL_FUNCTION(void, glCompileShader, GLuint shader);
DECLARE_GL_FUNCTION(void, glGetShaderiv, GLuint shader, GLShaderParameter pname, GLint *params);
DECLARE_GL_FUNCTION(void, glGetShaderInfoLog, GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
DECLARE_GL_FUNCTION(void, glDeleteShader, GLuint shader);
DECLARE_GL_FUNCTION(void, glAttachShader, GLuint program, GLuint shader);
DECLARE_GL_FUNCTION(void, glLinkProgram, GLuint program);
DECLARE_GL_FUNCTION(void, glGetProgramInfoLog, GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
DECLARE_GL_FUNCTION(void, glGetProgramiv, GLuint program, GLProgramParameter pname, GLint *params);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_430, void, glVertexAttribFormat, GLuint attribindex, GLint size, GLType type, GLboolean normalized, GLuint relativeoffset);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_430, void, glVertexAttribBinding, GLuint attribindex, GLuint bindingindex);
DECLARE_GL_FUNCTION(void, glEnableVertexAttribArray, GLuint index);
DECLARE_GL_FUNCTION(void, glEnableVertexAttribArrayARB, GLuint index);
DECLARE_GL_FUNCTION(void, glUseProgram, GLuint program);
DECLARE_GL_FUNCTION(void, glColorMaski, GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a);
DECLARE_GL_FUNCTION(void, glTexImage2DMultisample, GLTextureTarget target, GLsizei samples, GLFormat internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations);
DECLARE_GL_FUNCTION(void, glTexImage3DMultisample, GLTextureTarget target, GLsizei samples, GLFormat internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
DECLARE_GL_FUNCTION(void, glGenerateMipmap, GLTextureTarget target);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_TEXTURE_BINDLESS, void, glMakeTextureHandleNonResidentARB, GLuint64 handle);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_TEXTURE_BINDLESS, void, glTextureParameteriEXT, GLuint texture, GLTextureTarget target, GLTextureParameter pname, GLint param);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_TEXTURE_BINDLESS, void, glTextureParameterivEXT, GLuint texture, GLTextureTarget target, GLTextureParameter pname, const GLint *params);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_TEXTURE_BINDLESS, void, glTextureParameterfEXT, GLuint texture, GLTextureTarget target, GLTextureParameter pname, GLfloat param);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_TEXTURE_BINDLESS, void, glTextureParameterfvEXT, GLuint texture, GLTextureTarget target, GLTextureParameter pname, const GLfloat *params);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_TEXTURE_BINDLESS, GLuint64, glGetTextureHandleARB, GLuint texture);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_TEXTURE_BINDLESS, GLuint64, glGetTextureSamplerHandleARB, GLuint texture, GLuint sampler);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_TEXTURE_BINDLESS, void, glMakeTextureHandleResidentARB, GLuint64 handle);
DECLARE_GL_FUNCTION_OPTIONAL(TEMPEST_GL_CAPS_440, void, glBufferStorage, GLBufferTarget target, GLsizeiptr size, const void *data, GLbitfield flags);
DECLARE_GL_FUNCTION(void *, glMapBufferRange, GLBufferTarget target, GLintptr offset, GLsizeiptr length, GLbitfield access);
DECLARE_GL_FUNCTION(void, glVertexAttribPointer, GLuint index, GLint size, GLType type, GLboolean normalized, GLsizei stride, const void *pointer);

#ifndef TEMPEST_EXTRACT_FUNCTIONS
#ifdef _WIN32
#   define GL_GET_PROC_ADDRESS(func) wglGetProcAddress(func)
#elif defined(LINUX)
#   define GL_GET_PROC_ADDRESS(func) glXGetProcAddress(reinterpret_cast<const GLubyte*>(func))
#else
#	error "Unsupported platform"
#endif

#define GL_LIB_LOAD_FUNCTION(func) if(!LoadGLLibFunction(m_GLLib, TO_STRING(func), func)) return false;

template<typename TFunc>
bool LoadGLLibFunction(Library& gllib, const char* name, TFunc& func)
{
    func = reinterpret_cast<TFunc>(gllib.getProcAddress(name));
    return func != nullptr;
}

#define GL_LOAD_FUNCTION(func) if(!LoadGLFunction(m_GLLib, TO_STRING(func), func)) return false;
#define GL_LOAD_FUNCTION_OPTIONAL(caps, func) if(!LoadGLFunction(m_GLLib, TO_STRING(func), func)) GLCaps &= ~caps;

template<class TFunc>
bool LoadGLFunction(Library& gllib, const char* name, TFunc& func)
{
    func = reinterpret_cast<TFunc>(GL_GET_PROC_ADDRESS(name));
    if(!func)
    {
        func = reinterpret_cast<TFunc>(gllib.getProcAddress(name)); \
        return func != nullptr;
    }
    return true;
}
}
#endif

#endif /* GL_LIBRARY_HH */
