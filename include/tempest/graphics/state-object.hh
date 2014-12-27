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

#include "tempest/utils/types.hh"

namespace Tempest
{
enum class FillModeType: uint8
{
    Wireframe,
    Solid
};

enum class CullModeType: uint8
{
    FrontAndBack,
    Front,
    Back
};

enum class FrontFaceModeType: uint8
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
    int32               DepthBias            = 0;
    float               SlopeScaledDepthBias = 0.0f;
    uint32              MiscModes            = 0;
};

enum MiscBlendModesType
{
    TEMPEST_ALPHA_TO_COVERAGE_ENABLE = 1 << 0,
    TEMPEST_INDEPENDENT_BLEND_ENABLE = 1 << 1,
};

enum class BlendFactorType: uint16
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

enum class BlendOperationType: uint16
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
    uint8               RenderTargetWriteMask = 0xFF;
};

struct BlendStates
{
    uint32                  MiscModes = 0;
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
    uint8                   StencilReadMask  = 0xFF;
    uint8                   StencilWriteMask = 0xFF;
    uint8                   StencilRef       = 0;
    StencilOperationStates  FrontFace,
                            BackFace;
};
}