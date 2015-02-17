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

#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-state-object.hh"
#include "tempest/graphics/opengl-backend/gl-input-layout.hh"
#include "tempest/graphics/opengl-backend/gl-shader.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"
#include "tempest/graphics/state-object.hh"
#include "tempest/utils/assert.hh"

namespace Tempest
{
static GLFillMode TranslateFillMode(FillModeType fill_mode)
{
    switch(fill_mode)
    {
    default: TGE_ASSERT(false, "Unknown fill mode");
    case FillModeType::Solid: return GLFillMode::GL_FILL;
    case FillModeType::Wireframe: return GLFillMode::GL_LINE;
    }
}

static GLFaceMode TranslateCullMode(CullModeType cull_mode)
{
    switch(cull_mode)
    {
    default: TGE_ASSERT(false, "Unknown cull mode");
    case CullModeType::FrontAndBack: return GLFaceMode::GL_FRONT_AND_BACK;
    case CullModeType::Front: return GLFaceMode::GL_FRONT;
    case CullModeType::Back: return GLFaceMode::GL_BACK;
    }
}

static GLOrderMode TranslateFrontFaceMode(FrontFaceModeType front_face_mode)
{
    switch(front_face_mode)
    {
    default: TGE_ASSERT(false, "Unknown front face mode");
    case FrontFaceModeType::Clockwise: return GLOrderMode::GL_CW;
    case FrontFaceModeType::CounterClockwise: return GLOrderMode::GL_CCW;
    }
}

void TranslateRasterizerStates(const RasterizerStates* raster_states, GLRasterizerStates* gl_raster_states)
{
    gl_raster_states->PolygonMode = TranslateFillMode(raster_states->FillMode);
    gl_raster_states->CullFace = TranslateCullMode(raster_states->CullMode);
    gl_raster_states->FrontFace = TranslateFrontFaceMode(raster_states->FrontFaceMode);    
    gl_raster_states->OffsetFactor = raster_states->SlopeScaledDepthBias;
    gl_raster_states->OffsetUnits = static_cast<GLfloat>(raster_states->DepthBias);
    gl_raster_states->MiscModes = raster_states->MiscModes;
}

static GLBlendFactorMode TranslateBlendFactor(BlendFactorType factor)
{
    switch(factor)
    {
    default: TGE_ASSERT(false, "Unknown blend mode");
    case BlendFactorType::Zero: return GLBlendFactorMode::GL_ZERO;
    case BlendFactorType::One: return GLBlendFactorMode::GL_ONE;
    case BlendFactorType::SrcColor: return GLBlendFactorMode::GL_SRC_COLOR;
    case BlendFactorType::InvSrcColor: return GLBlendFactorMode::GL_ONE_MINUS_SRC_COLOR;
    case BlendFactorType::SrcAlpha: return GLBlendFactorMode::GL_SRC_ALPHA;
    case BlendFactorType::InvSrcAlpha: return GLBlendFactorMode::GL_ONE_MINUS_SRC_ALPHA;
    case BlendFactorType::DstAlpha: return GLBlendFactorMode::GL_DST_ALPHA;
    case BlendFactorType::InvDstAlpha: return GLBlendFactorMode::GL_ONE_MINUS_DST_ALPHA;
    case BlendFactorType::DstColor: return GLBlendFactorMode::GL_DST_COLOR;
    case BlendFactorType::InvDstColor: return GLBlendFactorMode::GL_ONE_MINUS_DST_COLOR;
    case BlendFactorType::SrcAlphaSat: return GLBlendFactorMode::GL_SRC_ALPHA_SATURATE;
    case BlendFactorType::BlendFactor: return GLBlendFactorMode::GL_CONSTANT_COLOR;
    case BlendFactorType::InvBlendFactor: return  GLBlendFactorMode::GL_ONE_MINUS_CONSTANT_COLOR;
    case BlendFactorType::Src1Color: return GLBlendFactorMode::GL_SRC1_COLOR;
    case BlendFactorType::InvSrc1Color: return GLBlendFactorMode::GL_ONE_MINUS_SRC1_COLOR;
    case BlendFactorType::Src1Alpha: return GLBlendFactorMode::GL_SRC1_ALPHA;
    case BlendFactorType::InvSrc1Alpha: return GLBlendFactorMode::GL_ONE_MINUS_SRC1_ALPHA;
    }
}

static GLBlendEquationMode TranslateBlendEquation(BlendOperationType op)
{
    switch(op)
    {
    default: TGE_ASSERT(false, "Unknown blend operation");
    case BlendOperationType::Add: return GLBlendEquationMode::GL_FUNC_ADD;
    case BlendOperationType::Subtract: return GLBlendEquationMode::GL_FUNC_SUBTRACT;
    case BlendOperationType::RevSubtract: return GLBlendEquationMode::GL_FUNC_REVERSE_SUBTRACT;
    case BlendOperationType::Min: return GLBlendEquationMode::GL_MIN;
    case BlendOperationType::Max: return GLBlendEquationMode::GL_MAX;
    }
}

void TranslateBlendStates(const BlendStates* blend_states, GLBlendStates* gl_blend_states)
{
    size_t max_render_targets = blend_states->MiscModes & TEMPEST_INDEPENDENT_BLEND_ENABLE ? TGE_FIXED_ARRAY_SIZE(blend_states->RenderTargets) : 1;
    gl_blend_states->MiscModes = blend_states->MiscModes;
    for(size_t i = 0; i < max_render_targets; ++i)
    {
        auto& rt_states = blend_states->RenderTargets[i];
        auto& gl_rt_states = gl_blend_states->SeparateBlendStates[i];
        gl_rt_states.BlendEnable = rt_states.BlendEnable;
        gl_rt_states.SrcFactor = TranslateBlendFactor(rt_states.SrcBlend);
        gl_rt_states.DstFactor = TranslateBlendFactor(rt_states.DstBlend);
        gl_rt_states.SrcFactorAlpha = TranslateBlendFactor(rt_states.SrcBlendAlpha);
        gl_rt_states.DstFactorAlpha = TranslateBlendFactor(rt_states.DstBlendAlpha);
        gl_rt_states.BlendEquation = TranslateBlendEquation(rt_states.BlendOp);
        gl_rt_states.BlendAlphaEquation = TranslateBlendEquation(rt_states.BlendOpAlpha);
        gl_rt_states.ColorMask = rt_states.RenderTargetWriteMask;
    }
}

static GLStencilOpMode TranslateStencilOperation(StencilOperationType stencil_op)
{
    switch(stencil_op)
    {
    default: TGE_ASSERT(false, "Unknown stencil operation");
    case StencilOperationType::Keep: return GLStencilOpMode::GL_KEEP;
    case StencilOperationType::Zero: return GLStencilOpMode::GL_ZERO;
    case StencilOperationType::Replace: return GLStencilOpMode::GL_REPLACE;
    case StencilOperationType::IncrSat: return GLStencilOpMode::GL_INCR;
    case StencilOperationType::DecrSat: return GLStencilOpMode::GL_DECR;
    case StencilOperationType::Invert: return GLStencilOpMode::GL_INVERT;
    case StencilOperationType::Incr: return GLStencilOpMode::GL_INCR_WRAP;
    case StencilOperationType::Decr: return GLStencilOpMode::GL_DECR_WRAP;
    }
}

static void TranslateStencilFaceOperations(const StencilOperationStates* stencil_op_states, GLDepthStencilOperationStates* gl_stencil_op_states)
{
    gl_stencil_op_states->StencilDepthFailOperation = TranslateStencilOperation(stencil_op_states->StencilDepthFailOperation);
    gl_stencil_op_states->StencilFailOperation = TranslateStencilOperation(stencil_op_states->StencilFailOperation);
    gl_stencil_op_states->StencilPassOperation = TranslateStencilOperation(stencil_op_states->StencilPassOperation);
    gl_stencil_op_states->StencilFunction = TranslateComparisonFunction(stencil_op_states->StencilFunction);
}

void TranslateDepthStencilStates(const DepthStencilStates* depth_stencil_states, GLDepthStencilStates* gl_depth_stencil_states)
{
    gl_depth_stencil_states->DepthTestEnable = depth_stencil_states->DepthTestEnable;
    gl_depth_stencil_states->DepthWriteEnable = depth_stencil_states->DepthWriteEnable;
    gl_depth_stencil_states->DepthFunction = TranslateComparisonFunction(depth_stencil_states->DepthFunction);
    gl_depth_stencil_states->StencilEnable = depth_stencil_states->StencilEnable;
    gl_depth_stencil_states->StencilReadMask = depth_stencil_states->StencilReadMask;
    gl_depth_stencil_states->StencilWriteMask = depth_stencil_states->StencilWriteMask;
    gl_depth_stencil_states->StencilRef = depth_stencil_states->StencilRef;
    TranslateStencilFaceOperations(&depth_stencil_states->FrontFace, &gl_depth_stencil_states->FrontFace);
    TranslateStencilFaceOperations(&depth_stencil_states->BackFace, &gl_depth_stencil_states->BackFace);
}

GLStateObject::GLStateObject(const GLShaderProgram* shader_prog,
                             DrawModes prim_type,
                             const GLRasterizerStates* rasterizer_states,
                             const GLBlendStates* blend_states,
                             const GLDepthStencilStates* depth_stencil_states)
    :   m_ShaderProgram(shader_prog),
        m_PrimitiveType(prim_type),
        m_RasterStates(rasterizer_states),
        m_BlendStates(blend_states),
        m_DepthStencilStates(depth_stencil_states)
{
}

bool operator==(const GLRasterizerStates& lhs, const GLRasterizerStates& rhs)
{
    return lhs.PolygonMode == rhs.PolygonMode &&
           lhs.CullFace == rhs.CullFace &&
           lhs.FrontFace == rhs.FrontFace &&
           lhs.OffsetFactor == rhs.OffsetFactor &&
           lhs.OffsetUnits == rhs.OffsetUnits &&
           lhs.MiscModes == rhs.MiscModes;
}

bool operator==(const GLBlendStates& lhs, const GLBlendStates& rhs)
{
    if(lhs.MiscModes != rhs.MiscModes)
       return false;

    size_t max_render_targets = lhs.MiscModes & TEMPEST_INDEPENDENT_BLEND_ENABLE ? TGE_FIXED_ARRAY_SIZE(GLBlendStates().SeparateBlendStates) : 1;
    for(size_t i = 0; i < max_render_targets; ++i)
    {
        auto& blend_states1 = lhs.SeparateBlendStates[i];
        auto& blend_states2 = rhs.SeparateBlendStates[i];
        if(blend_states1.BlendEnable != blend_states2.BlendEnable ||
           blend_states1.ColorMask != blend_states2.ColorMask)
            return false;
        // Don't bother with the rest of the stuff if blending is disabled.
        if(blend_states1.BlendEnable == GL_FALSE)
            continue;
        if(blend_states1.SrcFactor != blend_states2.SrcFactor ||
           blend_states1.DstFactor != blend_states2.DstFactor ||
           blend_states1.BlendEquation != blend_states2.BlendEquation ||
           blend_states1.SrcFactorAlpha != blend_states2.SrcFactorAlpha ||
           blend_states1.DstFactorAlpha != blend_states2.DstFactorAlpha ||
           blend_states1.BlendAlphaEquation != blend_states2.BlendAlphaEquation)
            return false;
    }
    return true;
}

bool operator==(const GLDepthStencilStates& lhs, const GLDepthStencilStates& rhs)
{
    if(lhs.DepthTestEnable != rhs.DepthTestEnable ||
       lhs.DepthWriteEnable != rhs.DepthWriteEnable)
       return false;
    if(lhs.DepthTestEnable == GL_TRUE &&
       lhs.DepthFunction != rhs.DepthFunction)
       return false;
    if(lhs.StencilEnable != rhs.StencilEnable)
        return false;
    if(lhs.StencilEnable == GL_TRUE)
    {
        auto cmp_face = [](const GLDepthStencilOperationStates& lhs, const GLDepthStencilOperationStates& rhs)
        {
            return lhs.StencilDepthFailOperation != rhs.StencilDepthFailOperation ||
                   lhs.StencilFailOperation != rhs.StencilFailOperation ||
                   lhs.StencilFunction != rhs.StencilFunction ||
                   lhs.StencilPassOperation != rhs.StencilPassOperation;
        };
        if(lhs.StencilReadMask != rhs.StencilReadMask ||
           lhs.StencilWriteMask != rhs.StencilWriteMask ||
           lhs.StencilRef != rhs.StencilRef ||
           cmp_face(lhs.FrontFace, rhs.FrontFace) ||
           cmp_face(lhs.BackFace, rhs.BackFace))
            return false;
    }
    return true;
}

bool GLStateObject::operator==(const GLStateObject& state_obj) const
{
    return m_ShaderProgram == state_obj.m_ShaderProgram &&
           m_PrimitiveType == state_obj.m_PrimitiveType &&
           m_RasterStates == state_obj.m_RasterStates &&
           m_BlendStates == state_obj.m_BlendStates &&
           m_DepthStencilStates == state_obj.m_DepthStencilStates;
}

void SetupState(uint32 mode_diff, uint32 misc_states, uint32 state, GLCapabilityMode gl_state)
{
    if(mode_diff & state)
    {
        if(misc_states & state)
        {
            glEnable(gl_state);
        }
        else
        {
            glDisable(gl_state);
        }
    }
}

const GLInputLayout* GLStateObject::getInputLayout() const { return m_ShaderProgram->getInputLayout(); }

void GLStateObject::setup(const GLStateObject* prev_state, GLBufferTableEntry* buffer_table) const
{
    if(prev_state == nullptr)
    {
        m_ShaderProgram->bind(buffer_table);
                
        auto* cur_rast_state = m_RasterStates;
        glPolygonMode(GLFaceMode::GL_FRONT_AND_BACK, cur_rast_state->PolygonMode);
        glCullFace(cur_rast_state->CullFace);
        glFrontFace(cur_rast_state->FrontFace);
        glPolygonOffset(cur_rast_state->OffsetFactor, cur_rast_state->OffsetUnits);
        uint32 mode_diff = ~0;
        SetupState(mode_diff, cur_rast_state->MiscModes, TEMPEST_DEPTH_CLIP_ENABLE, GLCapabilityMode::GL_DEPTH_CLAMP);
        SetupState(mode_diff, cur_rast_state->MiscModes, TEMPEST_SCISSOR_ENABLE, GLCapabilityMode::GL_SCISSOR_TEST);
        SetupState(mode_diff, cur_rast_state->MiscModes, TEMPEST_MULTISAMPLE_ENABLE, GLCapabilityMode::GL_MULTISAMPLE);
        SetupState(mode_diff, cur_rast_state->MiscModes, TEMPEST_ANTIALIASED_LINE_ENABLE, GLCapabilityMode::GL_LINE_SMOOTH);
        
        auto* cur_blend_state = m_BlendStates;
        SetupState(mode_diff, cur_blend_state->MiscModes, TEMPEST_ALPHA_TO_COVERAGE_ENABLE, GLCapabilityMode::GL_SAMPLE_ALPHA_TO_COVERAGE);
        if(cur_blend_state->MiscModes & TEMPEST_INDEPENDENT_BLEND_ENABLE)
        {
            for(GLuint i = 0; i < TGE_FIXED_ARRAY_SIZE(GLBlendStates().SeparateBlendStates); ++i)
            {
                auto& cur_rt_blend_states = cur_blend_state->SeparateBlendStates[i];
                if(cur_rt_blend_states.BlendEnable == GL_TRUE)
                {
                    glEnablei(GLCapabilityMode::GL_BLEND, i);
                    glBlendFunci(i, cur_rt_blend_states.SrcFactor, cur_rt_blend_states.DstFactor);
                    glBlendEquationi(i, cur_rt_blend_states.BlendEquation);
                    glBlendFunci(i, cur_rt_blend_states.SrcFactorAlpha, cur_rt_blend_states.DstFactorAlpha);
                    glBlendEquationi(i, cur_rt_blend_states.BlendAlphaEquation);
                }
                else
                {
                    glDisablei(GLCapabilityMode::GL_BLEND, i);
                }
                glColorMaski(i, cur_rt_blend_states.ColorMask & (1 << 0),
                             cur_rt_blend_states.ColorMask & (1 << 1),
                             cur_rt_blend_states.ColorMask & (1 << 2),
                             cur_rt_blend_states.ColorMask & (1 << 3));
            }
        }
        else
        {
            auto& cur_rt_blend_states = cur_blend_state->SeparateBlendStates[0];
            if(cur_rt_blend_states.BlendEnable == GL_TRUE)
            {
                glEnable(GLCapabilityMode::GL_BLEND);
                glBlendFunc(cur_rt_blend_states.SrcFactor, cur_rt_blend_states.DstFactor);
                glBlendEquation(cur_rt_blend_states.BlendEquation);
                glBlendFunc(cur_rt_blend_states.SrcFactorAlpha, cur_rt_blend_states.DstFactorAlpha);
                glBlendEquation(cur_rt_blend_states.BlendAlphaEquation);
            }
            else
            {
                glDisable(GLCapabilityMode::GL_BLEND);
            }
            glColorMask(cur_rt_blend_states.ColorMask & (1 << 0),
                        cur_rt_blend_states.ColorMask & (1 << 1),
                        cur_rt_blend_states.ColorMask & (1 << 2),
                        cur_rt_blend_states.ColorMask & (1 << 3));
        }

        auto* cur_ds_states = m_DepthStencilStates;
        auto* enable_depth_func = cur_ds_states->DepthTestEnable ? glEnable : glDisable;
        enable_depth_func(GLCapabilityMode::GL_DEPTH_TEST);
        glDepthMask(cur_ds_states->DepthWriteEnable);
        if(cur_ds_states->DepthTestEnable == GL_TRUE)
        {
            glDepthFunc(cur_ds_states->DepthFunction);
        }
        auto* enable_stencil_func = cur_ds_states->StencilEnable ? glEnable : glDisable;
        enable_stencil_func(GLCapabilityMode::GL_STENCIL_TEST);
        if(cur_ds_states->StencilEnable == GL_TRUE)
        {
            auto setup_face = [](uint8 mask, uint8 ref, const GLDepthStencilOperationStates& cur_ds_states)
            {
                glStencilFunc(cur_ds_states.StencilFunction, ref, mask);
                glStencilOp(cur_ds_states.StencilFailOperation,
                            cur_ds_states.StencilDepthFailOperation,
                            cur_ds_states.StencilPassOperation);
            };
            glStencilMask(cur_ds_states->StencilWriteMask);
            setup_face(cur_ds_states->StencilReadMask, cur_ds_states->StencilRef, cur_ds_states->FrontFace);
            setup_face(cur_ds_states->StencilReadMask, cur_ds_states->StencilRef, cur_ds_states->BackFace);
        }

        return;
    }

	if(m_ShaderProgram != prev_state->m_ShaderProgram)
	{
        m_ShaderProgram->bind(buffer_table);
	}

    if(m_RasterStates != prev_state->m_RasterStates)
    {
        auto* cur_rast_state = m_RasterStates;
        auto* old_rast_state = prev_state->m_RasterStates;
        if(cur_rast_state->PolygonMode != old_rast_state->PolygonMode)
        {
            glPolygonMode(GLFaceMode::GL_FRONT_AND_BACK, cur_rast_state->PolygonMode);
        }
        if(cur_rast_state->CullFace != old_rast_state->CullFace)
        {
            glCullFace(cur_rast_state->CullFace);
        }
        if(cur_rast_state->FrontFace != old_rast_state->FrontFace)
        {
            glFrontFace(cur_rast_state->FrontFace);
        }
        if(cur_rast_state->OffsetFactor != old_rast_state->OffsetFactor ||
           cur_rast_state->OffsetUnits != old_rast_state->OffsetUnits)
        {
            glPolygonOffset(cur_rast_state->OffsetFactor, cur_rast_state->OffsetUnits);
        }
        auto mode_diff = cur_rast_state->MiscModes ^ old_rast_state->MiscModes;
        SetupState(mode_diff, cur_rast_state->MiscModes, TEMPEST_DEPTH_CLIP_ENABLE, GLCapabilityMode::GL_DEPTH_CLAMP);
        SetupState(mode_diff, cur_rast_state->MiscModes, TEMPEST_SCISSOR_ENABLE, GLCapabilityMode::GL_SCISSOR_TEST);
        SetupState(mode_diff, cur_rast_state->MiscModes, TEMPEST_MULTISAMPLE_ENABLE, GLCapabilityMode::GL_MULTISAMPLE);
        SetupState(mode_diff, cur_rast_state->MiscModes, TEMPEST_ANTIALIASED_LINE_ENABLE, GLCapabilityMode::GL_LINE_SMOOTH);
    }

    if(m_BlendStates != prev_state->m_BlendStates)
    {
        auto* cur_blend_state = m_BlendStates;
        auto* old_blend_state = prev_state->m_BlendStates;
        auto mode_diff = cur_blend_state->MiscModes ^ old_blend_state->MiscModes;
        SetupState(mode_diff, cur_blend_state->MiscModes, TEMPEST_ALPHA_TO_COVERAGE_ENABLE, GLCapabilityMode::GL_SAMPLE_ALPHA_TO_COVERAGE);
        
        if(mode_diff & TEMPEST_INDEPENDENT_BLEND_ENABLE)
        {
            if(cur_blend_state->MiscModes & TEMPEST_INDEPENDENT_BLEND_ENABLE)
            {
                for(GLuint i = 0; i < TGE_FIXED_ARRAY_SIZE(GLBlendStates().SeparateBlendStates); ++i)
                {
                    auto& cur_rt_blend_states = cur_blend_state->SeparateBlendStates[i];
                    if(cur_rt_blend_states.BlendEnable == GL_TRUE)
                    {
                        glEnablei(GLCapabilityMode::GL_BLEND, i);
                        glBlendFunci(i, cur_rt_blend_states.SrcFactor, cur_rt_blend_states.DstFactor);
                        glBlendEquationi(i, cur_rt_blend_states.BlendEquation);
                        glBlendFunci(i, cur_rt_blend_states.SrcFactorAlpha, cur_rt_blend_states.DstFactorAlpha);
                        glBlendEquationi(i, cur_rt_blend_states.BlendAlphaEquation);
                    }
                    else
                    {
                        glDisablei(GLCapabilityMode::GL_BLEND, i);
                    }
                    glColorMaski(i, cur_rt_blend_states.ColorMask & (1 << 0),
                                    cur_rt_blend_states.ColorMask & (1 << 1),
                                    cur_rt_blend_states.ColorMask & (1 << 2),
                                    cur_rt_blend_states.ColorMask & (1 << 3));
                }
            }
            else
            {
                auto& cur_rt_blend_states = cur_blend_state->SeparateBlendStates[0];
                if(cur_rt_blend_states.BlendEnable == GL_TRUE)
                {
                    glEnable(GLCapabilityMode::GL_BLEND);
                    glBlendFunc(cur_rt_blend_states.SrcFactor, cur_rt_blend_states.DstFactor);
                    glBlendEquation(cur_rt_blend_states.BlendEquation);
                    glBlendFunc(cur_rt_blend_states.SrcFactorAlpha, cur_rt_blend_states.DstFactorAlpha);
                    glBlendEquation(cur_rt_blend_states.BlendAlphaEquation);
                }
                else
                {
                    glDisable(GLCapabilityMode::GL_BLEND);
                }
                glColorMask(cur_rt_blend_states.ColorMask & (1 << 0),
                            cur_rt_blend_states.ColorMask & (1 << 1),
                            cur_rt_blend_states.ColorMask & (1 << 2),
                            cur_rt_blend_states.ColorMask & (1 << 3));
            }
        }
        else
        {
            if(cur_blend_state->MiscModes & TEMPEST_INDEPENDENT_BLEND_ENABLE)
            {
                for(GLuint i = 0; i < TGE_FIXED_ARRAY_SIZE(GLBlendStates().SeparateBlendStates); ++i)
                {
                    auto& blend_states1 = cur_blend_state->SeparateBlendStates[i];
                    auto& blend_states2 = old_blend_state->SeparateBlendStates[i];
                    if(blend_states1.BlendEnable != blend_states2.BlendEnable)
                    {
                        auto* enable_blend_func = blend_states1.BlendEnable ? glEnablei : glDisablei;
                        enable_blend_func(GLCapabilityMode::GL_BLEND, i);
                    }
                    if(blend_states1.ColorMask != blend_states2.ColorMask)
                    {
                        glColorMaski(i, blend_states1.ColorMask & (1 << 0),
                                        blend_states1.ColorMask & (1 << 1),
                                        blend_states1.ColorMask & (1 << 2),
                                        blend_states1.ColorMask & (1 << 3));
                    }
                    // Don't bother with the rest of the stuff if blending is disabled.
                    if(blend_states1.BlendEnable == GL_FALSE)
                        continue;
                    if(blend_states1.SrcFactor != blend_states2.SrcFactor ||
                       blend_states1.DstFactor != blend_states2.DstFactor)
                    {
                        glBlendFunci(i, blend_states1.SrcFactor, blend_states1.DstFactor);
                    }
                    if(blend_states1.BlendEquation != blend_states2.BlendEquation)
                    {
                        glBlendEquationi(i, blend_states1.BlendEquation);
                    }
                    if(blend_states1.SrcFactorAlpha != blend_states2.SrcFactorAlpha ||
                       blend_states1.DstFactorAlpha != blend_states2.DstFactorAlpha)
                    {
                        glBlendFunci(i, blend_states1.SrcFactorAlpha, blend_states1.DstFactorAlpha);
                    }
                    if(blend_states1.BlendAlphaEquation != blend_states2.BlendAlphaEquation)
                    {
                        glBlendEquationi(i, blend_states1.BlendAlphaEquation);
                    }
                }
            }
            else
            {
                auto& blend_states1 = cur_blend_state->SeparateBlendStates[0];
                auto& blend_states2 = old_blend_state->SeparateBlendStates[0];
                if(blend_states1.BlendEnable != blend_states2.BlendEnable)
                {
                    auto* enable_blend_func = blend_states1.BlendEnable ? glEnable : glDisable;
                    enable_blend_func(GLCapabilityMode::GL_BLEND);
                }
                if(blend_states1.ColorMask != blend_states2.ColorMask)
                {
                    glColorMask(blend_states1.ColorMask & (1 << 0),
                                blend_states1.ColorMask & (1 << 1),
                                blend_states1.ColorMask & (1 << 2),
                                blend_states1.ColorMask & (1 << 3));
                }
                // Don't bother with the rest of the stuff if blending is disabled.
                if(blend_states1.BlendEnable == GL_TRUE)
                {
                    if(blend_states1.SrcFactor != blend_states2.SrcFactor ||
                       blend_states1.DstFactor != blend_states2.DstFactor)
                    {
                        glBlendFunc(blend_states1.SrcFactor, blend_states1.DstFactor);
                    }
                    if(blend_states1.BlendEquation != blend_states2.BlendEquation)
                    {
                        glBlendEquation(blend_states1.BlendEquation);
                    }
                    if(blend_states1.SrcFactorAlpha != blend_states2.SrcFactorAlpha ||
                       blend_states1.DstFactorAlpha != blend_states2.DstFactorAlpha)
                    {
                        glBlendFunc(blend_states1.SrcFactorAlpha, blend_states1.DstFactorAlpha);
                    }
                    if(blend_states1.BlendAlphaEquation != blend_states2.BlendAlphaEquation)
                    {
                        glBlendEquation(blend_states1.BlendAlphaEquation);
                    }
                }
            }
        }
    }

    if(m_DepthStencilStates != prev_state->m_DepthStencilStates)
    {
        auto* cur_ds_states = m_DepthStencilStates;
        auto* old_ds_states = prev_state->m_DepthStencilStates;
        if(cur_ds_states->DepthTestEnable != old_ds_states->DepthTestEnable)
        {
            auto* enable_depth_func = cur_ds_states->DepthTestEnable ? glEnable : glDisable;
            enable_depth_func(GLCapabilityMode::GL_DEPTH_TEST);
        }
        if(cur_ds_states->DepthWriteEnable != old_ds_states->DepthWriteEnable)
        {
            glDepthMask(cur_ds_states->DepthWriteEnable);
        }
        if(cur_ds_states->DepthTestEnable == GL_TRUE &&
           cur_ds_states->DepthFunction != old_ds_states->DepthFunction)
        {
            glDepthFunc(cur_ds_states->DepthFunction);
        }
        if(cur_ds_states->StencilEnable != old_ds_states->StencilEnable)
        {
            auto* enable_stencil_func = cur_ds_states->StencilEnable ? glEnable : glDisable;
            enable_stencil_func(GLCapabilityMode::GL_STENCIL_TEST);
        }
        if(cur_ds_states->StencilEnable == GL_TRUE)
        {
            auto setup_face = [](bool diff_states, uint8 mask, uint8 ref, const GLDepthStencilOperationStates& cur_ds_states, const GLDepthStencilOperationStates& old_ds_states)
            {
                if(diff_states || cur_ds_states.StencilFunction != old_ds_states.StencilFunction)
                {
                    glStencilFunc(cur_ds_states.StencilFunction, ref, mask);
                }
                if(cur_ds_states.StencilDepthFailOperation != old_ds_states.StencilDepthFailOperation ||
                   cur_ds_states.StencilFailOperation != old_ds_states.StencilFailOperation ||
                   cur_ds_states.StencilPassOperation != old_ds_states.StencilPassOperation)
                {
                    glStencilOp(cur_ds_states.StencilFailOperation,
                                cur_ds_states.StencilDepthFailOperation,
                                cur_ds_states.StencilPassOperation);
                }
            };
            if(cur_ds_states->StencilWriteMask != old_ds_states->StencilWriteMask)
            {
                glStencilMask(cur_ds_states->StencilWriteMask);
            }
            bool diff_states = cur_ds_states->StencilReadMask != old_ds_states->StencilReadMask ||
                               cur_ds_states->StencilRef != old_ds_states->StencilRef;
            setup_face(diff_states, cur_ds_states->StencilReadMask, cur_ds_states->StencilRef, cur_ds_states->FrontFace, old_ds_states->FrontFace);
            setup_face(diff_states, cur_ds_states->StencilReadMask, cur_ds_states->StencilRef, cur_ds_states->BackFace, old_ds_states->BackFace);
        }
    }
}
}   