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

#include "carina/common/exception.hh"

#include "carina/dx-renderer.hh"

#include <cassert>
#include <algorithm>

namespace Tempest
{
static DXGI_FORMAT TranslateIndicesType(IndicesType indices_type)
{
    switch(indices_type)
    {
    default: assert(!"Unknown indices type"); return (DXGI_FORMAT)0;
    case IT_UBYTE: return DXGI_FORMAT_R8_UINT;
    case IT_USHORT: return DXGI_FORMAT_R16_UINT;
    case IT_UINT: return DXGI_FORMAT_R32_UINT;
    }
}

DXGI_FORMAT D3DTranslateDataFormat(DataFormat vtype)
{
    switch(vtype)
    {
    default: assert(!"Unknown variable type"); return (DXGI_FORMAT)0;
    case VT_R32F: return DXGI_FORMAT_R32_FLOAT;
    case VT_RG32F: return DXGI_FORMAT_R32G32_FLOAT;
    case VT_RGB32F: return DXGI_FORMAT_R32G32B32_FLOAT;
    case VT_RGBA32F: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case VT_R16F: return DXGI_FORMAT_R16_FLOAT;
    case VT_RG16F: return DXGI_FORMAT_R16G16_FLOAT;
//  case VT_RGB16F: return DXGI_FORMAT_R16G16B16_FLOAT;
    case VT_RGBA16F: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case VT_R32: return DXGI_FORMAT_R32_SINT;
    case VT_RG32: return DXGI_FORMAT_R32G32_SINT;
    case VT_RGB32: return DXGI_FORMAT_R32G32B32_SINT;
    case VT_RGBA32: return DXGI_FORMAT_R32G32B32A32_SINT;
    case VT_R16: return DXGI_FORMAT_R16_SINT;
    case VT_RG16: return DXGI_FORMAT_R16G16_SINT;
//  case VT_RGB16: return DXGI_FORMAT_R16G16B16_SINT;
    case VT_RGBA16: return DXGI_FORMAT_R16G16B16A16_SINT;
    case VT_R8: return DXGI_FORMAT_R8_SINT;
    case VT_RG8: return DXGI_FORMAT_R8G8_SINT;
//  case VT_RGB8: return DXGI_FORMAT_R8G8B8_SINT;
    case VT_RGBA8: return DXGI_FORMAT_R8G8B8A8_SINT;
    case VT_uR32: return DXGI_FORMAT_R32_UINT;
    case VT_uRG32: return DXGI_FORMAT_R32G32_UINT;
    case VT_uRGB32: return DXGI_FORMAT_R32G32B32_UINT;
    case VT_uRGBA32: return DXGI_FORMAT_R32G32B32A32_UINT;
    case VT_uR16: return DXGI_FORMAT_R16_UINT;
    case VT_uRG16: return DXGI_FORMAT_R16G16_UINT;
//  case VT_uRGB16: return DXGI_FORMAT_R16G16B16_UINT;
    case VT_uRGBA16: return DXGI_FORMAT_R16G16B16A16_UINT;
    case VT_uR8: return DXGI_FORMAT_R8_UINT;
    case VT_uRG8: return DXGI_FORMAT_R8G8_UINT;
//  case VT_uRGB8: return DXGI_FORMAT_R8G8B8_UINT;
    case VT_uRGBA8: return DXGI_FORMAT_R8G8B8A8_UINT;
    case VT_R16_SNORM: return DXGI_FORMAT_R16_SNORM;
    case VT_RG16_SNORM: return DXGI_FORMAT_R16G16_SNORM;
//  case VT_RGB16_SNORM: return DXGI_FORMAT_R16G16B16_SNORM;
    case VT_RGBA16_SNORM: return DXGI_FORMAT_R16G16B16A16_SNORM;
    case VT_R8_SNORM: return DXGI_FORMAT_R8_SNORM;
    case VT_RG8_SNORM: return DXGI_FORMAT_R8G8_SNORM;
//  case VT_RGB8_SNORM: return DXGI_FORMAT_R8G8B8_SNORM;
    case VT_RGBA8_SNORM: return DXGI_FORMAT_R8G8B8A8_SNORM;
    case VT_R16_UNORM: return DXGI_FORMAT_R16_UNORM;
    case VT_RG16_UNORM: return DXGI_FORMAT_R16G16_UNORM;
//  case VT_RGB16_UNORM: return DXGI_FORMAT_R16G16B16_UNORM;
    case VT_RGBA16_UNORM: return DXGI_FORMAT_R16G16B16A16_UNORM;
    case VT_R8_UNORM: return DXGI_FORMAT_R8_UNORM;
    case VT_RG8_UNORM: return DXGI_FORMAT_R8G8_UNORM;
//  case VT_RGB8_UNORM: return DXGI_FORMAT_R8G8B8_UNORM;
    case VT_RGBA8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case VT_D16: return DXGI_FORMAT_R16_TYPELESS;
    case VT_D24S8: return DXGI_FORMAT_R24G8_TYPELESS;
    case VT_D32: return DXGI_FORMAT_R32_TYPELESS;
    case VT_R10G10B10A2: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case VT_UINT_R10G10B10A2: return DXGI_FORMAT_R10G10B10A2_UINT;
    }
}

static D3D11_PRIMITIVE_TOPOLOGY TranslateDrawMode(DrawModes mode)
{
    switch(mode)
    {
    default: assert(!"Unknown draw topology"); return (D3D11_PRIMITIVE_TOPOLOGY)0;
    case TGE_POINT_LIST: return D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
    case TGE_LINE_STRIP: return D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP;
    case TGE_LINE_LIST: return D3D11_PRIMITIVE_TOPOLOGY_LINELIST;
    case TGE_TRIANGLE_STRIP: return D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    case TGE_TRIANGLE_LIST: return D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    case TGE_LINE_STRIP_ADJ: return D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ;
    case TGE_LINE_LIST_ADJ: return D3D11_PRIMITIVE_TOPOLOGY_LINELIST_ADJ;
    case TGE_TRIANGLE_STRIP_ADJ: return D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ;
    case TGE_TRIANGLE_LIST_ADJ: return D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ;
    }
}

DXRenderer::DXRenderer(ComRAII<ID3D11Device>& dev)
    :   m_Device(dev),
        m_ClearColor(0.0f, 0.0f, 0.0f, 0.0f)
{
    dev->GetImmediateContext(&m_DeviceContext);
    m_DeviceContext->OMGetRenderTargets(1, &m_RenderTargetView, &m_DepthStencilView);

	m_CurrentState = m_DefaultState = make_aligned_shared<DXStateObject>(*this, MiscStateDescription());
    m_CurrentBlendState = m_DefaultBlendState = make_aligned_shared<DXBlendStateObject>(*this, BlendStateDescription());
    m_CurrentDepthStencilState = m_DefaultDepthStencilState = make_aligned_shared<DXDepthStencilStateObject>(*this, DepthStencilStateDescription());
    m_DefaultState->setup(*this);
    m_DefaultBlendState->setup(*this);

    UINT support;
    assert(m_Device->CheckFormatSupport(DXGI_FORMAT_R32G32B32_FLOAT, &support) == S_OK &&
            (support & D3D11_FORMAT_SUPPORT_MIP) && (support & D3D11_FORMAT_SUPPORT_TEXTURE2D));

    m_PreferredShaderCompiler = make_aligned_shared<HLSLShaderCompiler>(*this);
    m_ShaderCompilers.push_back(m_PreferredShaderCompiler);
}

DXRenderer::~DXRenderer()
{
}

void DXRenderer::setScissorRect(size_t x, size_t y, size_t width, size_t height)
{
    size_t vx, vy;
    getViewport(vx, vy);
    RECT r = {
                static_cast<LONG>(x),
                static_cast<LONG>(vy - y - height),
                static_cast<LONG>(x + width),
                static_cast<LONG>(vy - y)
             };
    m_DeviceContext->RSSetScissorRects(1, &r);
}

InputLayoutPtr DXRenderer::createInputLayout(const ShaderProgramPtr& shader, const DynamicArray<InputElement>& arr)
{
    return make_aligned_shared<DXInputLayout>(*this, shader, arr);
}

void DXRenderer::setInputLayout(const InputLayoutConstPtr& layout)
{
	if(layout)
		static_cast<const DXInputLayout&>(*layout).setup(*this);
	else
		m_DeviceContext->IASetInputLayout(nullptr);
}

VideoBufferPtr DXRenderer::createVideoBuffer(VBType type, size_t size, const void* data, VBUsage usage)
{
    return make_aligned_shared<DXVideoBuffer>(*this, type, size, data, usage);
}

void DXRenderer::setVideoBuffer(const VideoBufferPtr& vb, size_t stride, size_t offset)
{
    if(vb)
        static_cast<const DXVideoBuffer&>(*vb).bindVideoBuffer(*this, stride, offset);
    else
        m_DeviceContext->IASetVertexBuffers(0, 0, nullptr, nullptr, nullptr);
}

void DXRenderer::setIndexBuffer(const VideoBufferPtr& vb, IndicesType itype, size_t offset)
{
    if(vb)
        static_cast<const DXVideoBuffer&>(*vb).bindIndexBuffer(*this, TranslateIndicesType(itype), offset);
    else
        m_DeviceContext->IASetIndexBuffer(nullptr, DXGI_FORMAT_UNKNOWN, 0);
}

void DXRenderer::draw(DrawModes mode, size_t first, size_t count)
{
    m_DeviceContext->IASetPrimitiveTopology(TranslateDrawMode(mode));
    m_Prog->apply();
    m_DeviceContext->Draw(count, first);
}

void DXRenderer::drawIndexed(DrawModes mode, size_t first, size_t count)
{
    m_DeviceContext->IASetPrimitiveTopology(TranslateDrawMode(mode));
    m_Prog->apply();
    m_DeviceContext->DrawIndexed(count, first, 0);
}

ShaderCompilerPtr DXRenderer::getPreferredShaderCompiler(const ShadingLanguagesList& shlang_list)
{
    return m_PreferredShaderCompiler;
}

ShaderCompilerPtr DXRenderer::getShaderCompiter(const string& shading_language)
{
	auto i = std::find_if(m_ShaderCompilers.begin(), m_ShaderCompilers.end(), [&shading_language](const ShaderCompilerPtr& sc) { return sc->getName() == shading_language; });
    if(i == m_ShaderCompilers.end())
        THROW_EXCEPTION("Unknown shader compiler: " + shading_language);
    return *i;
}

void DXRenderer::setPreferredShadingLanguage(const string& name)
{
    auto i = std::find_if(m_ShaderCompilers.begin(), m_ShaderCompilers.end(), [&name](const ShaderCompilerPtr& sc) { return sc->getName() == name; });
    if(i == m_ShaderCompilers.end())
        THROW_EXCEPTION("unknown shader compiler: " + name);
    m_PreferredShaderCompiler = *i;
}

void DXRenderer::useShaderProgram(const ShaderProgramPtr& prog)
{
    m_Prog = prog;
}

TexturePtr DXRenderer::createTexture(const TextureDescription& desc, const void* data, int flags)
{
    return make_aligned_shared<DXTexture>(*this, desc, data, flags);
}

void DXRenderer::resizeViewport(size_t w, size_t h)
{
    D3D11_VIEWPORT vp;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    vp.Height = static_cast<FLOAT>(h);
    vp.Width = static_cast<FLOAT>(w);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;

    m_DeviceContext->RSSetViewports(1, &vp);
}

void DXRenderer::getViewport(size_t& w, size_t& h)
{
    UINT num = 1;
    D3D11_VIEWPORT vp;
    m_DeviceContext->RSGetViewports(&num, &vp);
    w = static_cast<size_t>(vp.Width);
    h = static_cast<size_t>(vp.Height);
}

StateObjectPtr DXRenderer::createStateObject(const MiscStateDescription& desc)
{
    return make_aligned_shared<DXStateObject>(*this, desc);
}

void DXRenderer::setStateObject(const StateObjectPtr& state_obj)
{
    m_CurrentState = static_pointer_cast<DXStateObject>(state_obj);
    if(state_obj.get())
        m_CurrentState->setup(*this);
    else
        m_DefaultState->setup(*this);
}

StateObjectPtr DXRenderer::getStateObject()
{
    return m_CurrentState;
}

BlendStateObjectPtr DXRenderer::createBlendStateObject(const BlendStateDescription& desc)
{
    return make_aligned_shared<DXBlendStateObject>(*this, desc);
}

void DXRenderer::setBlendStateObject(const BlendStateObjectPtr& bstate_obj)
{
    m_CurrentBlendState = static_pointer_cast<DXBlendStateObject>(bstate_obj);
    if(bstate_obj.get())
        m_CurrentBlendState->setup(*this);
    else
        m_DefaultBlendState->setup(*this);
}

BlendStateObjectPtr DXRenderer::getBlendStateObject()
{
    return m_CurrentBlendState;
}

Vector4 DXRenderer::getClearColor() const
{
    return m_ClearColor;
}

void DXRenderer::setClearColor(const Vector4& color)
{
    m_ClearColor = color;
}

void DXRenderer::clearColorBuffer(size_t idx, const Vector4& color)
{
    if(!m_DrawFramebuffer.get() && idx)
        THROW_EXCEPTION("DirectX: Unknown buffer");
    m_DrawFramebuffer->clearBuffer(idx, color);
}

void DXRenderer::clearAll()
{
    if(m_DrawFramebuffer.get())
        m_DrawFramebuffer->clearAll(m_ClearColor);
    else
    {
        m_DeviceContext->ClearRenderTargetView(m_RenderTargetView, reinterpret_cast<float*>(&m_ClearColor));
        m_DeviceContext->ClearDepthStencilView(m_DepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
    }
}

void DXRenderer::clearDepthStencilBuffer(float depth, int stencil)
{
    if(m_DrawFramebuffer.get())
        m_DrawFramebuffer->clearDepthStencilBuffer(depth, stencil);
    else
        m_DeviceContext->ClearDepthStencilView(m_DepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
}

Matrix4 DXRenderer::createPerspectiveMatrix(float fovy, float aspect, float zNear, float zFar)
{
    float f = 1.0f / tan((fovy / 360.0f) * math_pi);
    return Matrix4(f / aspect,  0.0f,   0.0f,                           0.0f,
                   0.0f,        f,      0.0f,                           0.0f,
                   0.0f,        0.0f,   zFar/(zNear - zFar),           -1.0f,
                   0.0f,        0.0f,   (zNear*zFar)/(zNear - zFar),    0.0f);
}

Matrix4 DXRenderer::createOrthoMatrix(float left, float right, float bottom, float top, float _near, float _far)
{
    return Matrix4(2.0f/(right-left),           0.0f,                       0.0f,                       0.0f,
                   0.0f,                        2.0f/(top-bottom),          0.0f,                       0.0f,
                   0.0f,                        0.0f,                       1.0f/(_far-_near),          0.0f,
                   (right+left)/(left-right),   (top+bottom)/(bottom-top),  _near/(_near-_far),         1.0f);
}

void DXRenderer::_updateBuffers()
{
    m_DeviceContext->OMGetRenderTargets(1, &m_RenderTargetView, &m_DepthStencilView);
}

void DXRenderer::_releaseBuffers()
{
    m_RenderTargetView.release();
    m_DepthStencilView.release();
}

FramebufferPtr DXRenderer::createFramebuffer()
{
    return make_aligned_shared<DXFramebuffer>(*this);
}

void DXRenderer::bindFramebuffer(const FramebufferPtr& fb)
{
    m_ReadFramebuffer = m_DrawFramebuffer = static_pointer_cast<DXFramebuffer>(fb);
    if(fb)
        static_cast<DXFramebuffer&>(*fb).bind();
    else
        m_DeviceContext->OMSetRenderTargets(1, &m_RenderTargetView, m_DepthStencilView);
}

void DXRenderer::bindDrawFramebuffer(const FramebufferPtr& fb)
{
    m_DrawFramebuffer = static_pointer_cast<DXFramebuffer>(fb);
    if(fb)
        static_cast<DXFramebuffer&>(*fb).bind();
    else
        m_DeviceContext->OMSetRenderTargets(1, &m_RenderTargetView, m_DepthStencilView);
}

void DXRenderer::bindReadFramebuffer(const FramebufferPtr& fb)
{
    m_ReadFramebuffer = m_DrawFramebuffer;
}

void DXRenderer::blitFramebuffer(int buffers)
{
    if(m_ReadFramebuffer == m_DrawFramebuffer)
        THROW_EXCEPTION("the read buffer and the draw buffer must be different");
    ComRAII<ID3D11Resource> read_buffer,
                            draw_buffer;
    if(buffers & TGE_COLOR_BUFFER_BIT)
    {
        if(m_ReadFramebuffer.get())
            read_buffer = ComRAII<ID3D11Resource>(m_ReadFramebuffer->getColorAttachment(0)->getTexture());
        else
            m_RenderTargetView->GetResource(&read_buffer);

        if(m_DrawFramebuffer.get())
            draw_buffer = ComRAII<ID3D11Resource>(m_DrawFramebuffer->getColorAttachment(0)->getTexture());
        else
            m_RenderTargetView->GetResource(&draw_buffer);

        m_DeviceContext->CopyResource(draw_buffer, read_buffer);
    }
    if(buffers & TGE_DEPTH_BUFFER_BIT)
    {
        if(m_ReadFramebuffer.get())
            read_buffer = m_ReadFramebuffer->getDepthStencil()->getTexture();
        else
            m_DepthStencilView->GetResource(&read_buffer);

        if(m_DrawFramebuffer.get())
            draw_buffer = m_DrawFramebuffer->getDepthStencil()->getTexture();
        else
            m_DepthStencilView->GetResource(&draw_buffer);
        
        m_DeviceContext->CopyResource(draw_buffer, read_buffer);
    }
}

FramebufferPtr DXRenderer::getDrawFramebuffer()
{
    return m_DrawFramebuffer;
}

FramebufferPtr DXRenderer::getReadFramebuffer()
{
    return m_ReadFramebuffer;
}

DepthStencilStateObjectPtr DXRenderer::createDepthStencilStateObject(const DepthStencilStateDescription& desc)
{
    return make_aligned_shared<DXDepthStencilStateObject>(*this, desc);
}

void DXRenderer::setDepthStencilStateObject(const DepthStencilStateObjectPtr& dsstate_obj)
{
    m_CurrentDepthStencilState = static_pointer_cast<DXDepthStencilStateObject>(dsstate_obj);
    m_CurrentDepthStencilState->setup(*this);
}

DepthStencilStateObjectPtr DXRenderer::getDepthStencilStateObject()
{
    return m_CurrentDepthStencilState;
}

void DXRenderer::registerFileLoader(const FileLoaderPtr& include_loader)
{
    m_IncludeHandler.setFileLoader(include_loader);
}

DXIncludeProxyHandler& DXRenderer::getIncludeHandler()
{
    return m_IncludeHandler;
}
}
