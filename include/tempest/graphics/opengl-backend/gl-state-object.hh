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

#ifndef _GL_STATE_OBJECT_HH_
#define _GL_STATE_OBJECT_HH_

#include <cstdint>
#include "tempest/graphics/rendering-definitions.hh"

#include "tempest/graphics/opengl-backend/gl-library.hh"

namespace Tempest
{
class GLInputLayout;
class GLShaderProgram;
struct RasterizerStates;
struct BlendStates;
struct DepthStencilStates;

struct GLRasterizerStates
{
    GLFillMode          PolygonMode;
    GLFaceMode          CullFace;
    GLOrderMode         FrontFace;
    GLfloat             OffsetFactor;
    GLfloat             OffsetUnits;
    uint32_t            MiscModes;
};

void TranslateRasterizerStates(const RasterizerStates* raster_states, GLRasterizerStates* gl_raster_states);
bool operator==(const GLRasterizerStates& lhs, const GLRasterizerStates& rhs);

struct GLBlendSeparateStates
{
    bool                  BlendEnable;
    GLBlendFactorMode     SrcFactor,
                          DstFactor,
                          SrcFactorAlpha,
                          DstFactorAlpha;
    GLBlendEquationMode   BlendEquation,
                          BlendAlphaEquation;
    uint8_t               ColorMask;
};

struct GLBlendStates
{
    uint32_t              MiscModes;
    GLBlendSeparateStates SeparateBlendStates[8];
};

void TranslateBlendStates(const BlendStates* blend_states, GLBlendStates* gl_blend_states);
bool operator==(const GLBlendStates& lhs, const GLBlendStates& rhs);

struct GLDepthStencilOperationStates
{
    GLStencilOpMode       StencilFailOperation,
                          StencilDepthFailOperation,
                          StencilPassOperation;
    GLComparisonFunction  StencilFunction;
};

struct GLDepthStencilStates
{
    GLboolean             DepthTestEnable;
    GLboolean             DepthWriteEnable;
    GLComparisonFunction  DepthFunction;
    GLboolean             StencilEnable;
    GLuint                StencilReadMask;
    GLuint                StencilWriteMask;
    GLuint                StencilRef;
    GLDepthStencilOperationStates FrontFace,
                                  BackFace;
};

struct GLBufferTableEntry;

void TranslateDepthStencilStates(const DepthStencilStates* depth_stencil_states, GLDepthStencilStates* gl_depth_stencil_states);
bool operator==(const GLDepthStencilStates& lhs, const GLDepthStencilStates& rhs);

class GLStateObject
{
    const GLShaderProgram      *m_ShaderProgram;
    DrawModes                   m_PrimitiveType;
    const GLRasterizerStates   *m_RasterStates;
    const GLBlendStates        *m_BlendStates;
    const GLDepthStencilStates *m_DepthStencilStates;
public:
    GLStateObject(const GLShaderProgram* shader_prog,
                  DrawModes prim_type,
                  const GLRasterizerStates* rasterizer_states,
                  const GLBlendStates* blend_states,
                  const GLDepthStencilStates* depth_stencil_states);

    bool operator==(const GLStateObject&) const;

    const GLInputLayout* getInputLayout() const;
    DrawModes getPrimitiveType() { return m_PrimitiveType; }

    // prev_state is purely for optimization purposes.
    void setup(const GLStateObject* prev_state, GLBufferTableEntry* buffer_table) const;
};
}

#endif // _GL_STATE_OBJECT_HH_