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

#ifndef _TEMPEST_STATE_OBJECT_HH_
#define _TEMPEST_STATE_OBJECT_HH_

#include <cstdint>

namespace Tempest
{
enum class FillModeType: uint8_t
{
    Wireframe,
    Solid
};

enum class CullModeType: uint8_t
{
    FrontAndBack,
    Front,
    Back
};

enum class FrontFaceModeType: uint8_t
{
    CounterClockwise,
    Clockwise
};

enum MiscRasterizerModesType
{
    TEMPEST_DEPTH_CLIP_ENABLE       = 1 << 0,
    TEMPEST_SCISSOR_ENABLE          = 1 << 1,
    TEMPEST_MULTISAMPLE_ENABLE      = 1 << 2,
    TEMPEST_ANTIALIASED_LINE_ENABLE = 1 << 3
};

struct RasterizerStates
{
    FillModeType        FillMode             = FillModeType::Solid;
    CullModeType        CullMode             = CullModeType::Back;
    FrontFaceModeType   FrontFaceMode        = FrontFaceModeType::CounterClockwise;
    int32_t             DepthBias            = 0;
    float               SlopeScaledDepthBias = 0.0f;
    uint32_t            MiscModes            = 0;
};

enum MiscBlendModesType
{
    TEMPEST_ALPHA_TO_COVERAGE_ENABLE = 1 << 0,
    TEMPEST_INDEPENDENT_BLEND_ENABLE = 1 << 1,
};

enum class BlendFactorType: uint16_t
{
    Zero,
    One,
    SrcColor,
    InvSrcColor,
    SrcAlpha,
    InvSrcAlpha,
    DstAlpha,
    InvDstAlpha,
    DstColor,
    InvDstColor,
    SrcAlphaSat,
    BlendFactor,
    InvBlendFactor,
    Src1Color,
    InvSrc1Color,
    Src1Alpha,
    InvSrc1Alpha
};

enum class BlendOperationType: uint16_t
{
    Add,
    Subtract,
    RevSubtract,
    Min,
    Max
};

struct RenderTargetBlendStates
{
    bool                BlendEnable           = false;
    BlendFactorType     SrcBlend              = BlendFactorType::One,
                        DstBlend              = BlendFactorType::Zero;
    BlendOperationType  BlendOp               = BlendOperationType::Add;
    BlendFactorType     SrcBlendAlpha         = BlendFactorType::One,
                        DstBlendAlpha         = BlendFactorType::Zero;
    BlendOperationType  BlendOpAlpha          = BlendOperationType::Add;
    uint8_t             RenderTargetWriteMask = 0xFF;
};

struct BlendStates
{
    uint32_t                MiscModes = 0;
    RenderTargetBlendStates RenderTargets[8];
};

enum class StencilOperationType
{
    Keep,
    Zero,
    Replace,
    IncrSat,
    DecrSat,
    Invert,
    Incr,
    Decr
};

struct StencilOperationStates
{
    StencilOperationType   StencilFailOperation      = StencilOperationType::Keep,
                           StencilDepthFailOperation = StencilOperationType::Keep,
                           StencilPassOperation      = StencilOperationType::Keep;
    ComparisonFunction     StencilFunction           = ComparisonFunction::AlwaysPass;
};

struct DepthStencilStates
{
    bool                    DepthTestEnable  = false;
    bool                    DepthWriteEnable = false;
    ComparisonFunction      DepthFunction    = ComparisonFunction::Less;
    bool                    StencilEnable    = false;
    uint8_t                 StencilReadMask  = 0xFF;
    uint8_t                 StencilWriteMask = 0xFF;
    uint8_t                 StencilRef       = 0;
    StencilOperationStates  FrontFace,
                            BackFace;
};
}

#endif // _TEMPEST_STATE_OBJECT_HH_