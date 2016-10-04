/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2016 Zdravko Velinov
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

#include "tempest/graphics/scene-rasterizer-system.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/state-object.hh"

namespace Tempest
{
SceneRasterizerSystem::SceneRasterizerSystem(uint32_t image_width, uint32_t image_height, const Matrix4& view_proj_inv,
                                             PreferredWindow* window, PreferredBackend* backend, PreferredShaderCompiler* shader_compiler)
    :   m_TextureSlots(TotalSlots*4*sizeof(float))
{
    m_FrameTexture.Width = image_width;
    m_FrameTexture.Height = image_height;

    memset(m_TextureSlots.get(), 0, TotalSlots*4*sizeof(float));
    if(window && backend && shader_compiler)
    {
        m_Window = window;
        m_Backend = backend;
        m_ShaderCompiler = shader_compiler;
    }
    else
    {
        TGE_ASSERT(window == nullptr && backend == nullptr && shader_compiler == nullptr, "Broken pipeline");

        Tempest::WindowDescription wdesc;
        wdesc.Width = image_width;
        wdesc.Height = image_height;
        wdesc.Title = "Test window";
        m_SystemCleanup = Tempest::CreateSystemAndWindowSimple<Tempest::PreferredSystem>(wdesc); 
    
        m_Window = &m_SystemCleanup->Window;
        m_Backend = &m_SystemCleanup->Backend;
        m_ShaderCompiler = &m_SystemCleanup->ShaderCompiler;
    }

    m_Rasterizer = std::unique_ptr<SceneRasterizer>(new SceneRasterizer(m_Backend, m_ShaderCompiler, image_width, image_height, view_proj_inv));

    m_Shader = Tempest::CreateShader(m_ShaderCompiler, SOURCE_SHADING_DIR "/ray-trace-blit.tfx").release();
    TGE_ASSERT(m_Shader, "Failed to load ray trace backbuffer blit shader");

    auto rt_fmt = Tempest::DataFormat::RGBA8UNorm;
    Tempest::DepthStencilStates ds_state;
    ds_state.DepthTestEnable = false;
    ds_state.DepthWriteEnable = false;
    m_DSState = Tempest::CreateStateObject(m_Backend, &rt_fmt, 1, Tempest::DataFormat::Unknown, m_Shader, Tempest::DrawModes::TriangleList, nullptr, nullptr, &ds_state).release();

    Tempest::PreferredBackend::CommandBufferType::DrawBatchType blit_batch;
    blit_batch.VertexCount = 3;
    blit_batch.PipelineState = m_DSState;
    blit_batch.ResourceTable = nullptr;

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = 1;
    cmd_buffer_desc.ConstantsBufferSize = 1024;
    m_CommandBuffer = Tempest::CreateCommandBuffer(m_Backend, cmd_buffer_desc).release();
    m_CommandBuffer->enqueueBatch(blit_batch);
    m_CommandBuffer->prepareCommandBuffer();

    SubdirectoryFileLoader subdir_loader(SOURCE_SHADING_DIR);
    m_FpsCounter = std::unique_ptr<FpsCounter>(new FpsCounter(m_Backend, m_ShaderCompiler, &subdir_loader, 50.0f, static_cast<float>(image_width), static_cast<float>(image_height)));

    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;
    m_FpsCommandBuffer = m_Backend->createCommandBuffer(cmd_buffer_desc);


}

SceneRasterizerSystem::~SceneRasterizerSystem()
{
    m_Backend->destroyRenderResource(m_CommandBuffer);

    m_Backend->destroyRenderResource(m_DSState);

    m_ShaderCompiler->destroyRenderResource(m_Shader);
}
}