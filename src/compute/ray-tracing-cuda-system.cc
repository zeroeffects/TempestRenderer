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

#include "tempest/compute/ray-tracing-cuda-system.hh"

#include "driver_types.h"
#include "cuda_runtime_api.h"

namespace Tempest
{
RayTracingCudaSystem::RayTracingCudaSystem(uint32_t width, uint32_t height, const Matrix4& view_proj_inv, const RTSettings& settings,
                                           PreferredWindow* window, PreferredBackend* backend, PreferredShaderCompiler* shader_compiler)
    :   m_TextureSlots(TotalSlots*4*sizeof(float))
{
    m_Texture = nullptr;
    memset(&m_CudaTexture, 0, sizeof(m_CudaTexture));

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
        wdesc.Width = width;
        wdesc.Height = height;
        wdesc.Title = "Test window";
        m_SystemCleanup = Tempest::CreateSystemAndWindowSimple<Tempest::PreferredSystem>(wdesc); 
    
        m_Window = &m_SystemCleanup->Window;
        m_Backend = &m_SystemCleanup->Backend;
        m_ShaderCompiler = &m_SystemCleanup->ShaderCompiler;
    }

    m_FrameTexture.Width = width;
    m_FrameTexture.Height = height;
    m_FrameTexture.Format = Tempest::DataFormat::RGBA8UNorm;
    auto gpu_tex = m_Texture = m_Backend->createTexture(m_FrameTexture, Tempest::RESOURCE_DYNAMIC_DRAW);

    m_Backend->mapToCudaSurface(gpu_tex, cudaGraphicsRegisterFlagsSurfaceLoadStore, &m_CudaTexture);
    m_RayTracer = std::unique_ptr<RayTracerCuda>(new RayTracerCuda(&m_CudaTexture, view_proj_inv, settings));

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

    Tempest::SubdirectoryFileLoader subdir_loader(SOURCE_SHADING_DIR);
    m_FpsCounter = std::unique_ptr<FpsCounter>(new FpsCounter(m_Backend, m_ShaderCompiler, &subdir_loader, 50.0f, static_cast<float>(width), static_cast<float>(height)));

    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;
    m_FpsCommandBuffer = m_Backend->createCommandBuffer(cmd_buffer_desc);
}

RayTracingCudaSystem::~RayTracingCudaSystem()
{
    m_Backend->destroyRenderResource(m_Texture);
    if(m_CudaTexture.Surface)
    {
        m_Backend->unmapCudaSurface(&m_CudaTexture);
    }

    m_ShaderCompiler->destroyRenderResource(m_Shader);
    m_Backend->destroyRenderResource(m_DSState);
    m_Backend->destroyRenderResource(m_CommandBuffer);
    m_Backend->destroyRenderResource(m_FpsCommandBuffer);
}

void RayTracingCudaSystem::completeFrameAndRestart(uint32_t width, uint32_t height, const Matrix4& view_proj_inv)
{
    auto& next_tex = m_Texture;
    auto& next_cuda_tex = m_CudaTexture;
    if(next_cuda_tex.Texture.Description.Width != width ||
        next_cuda_tex.Texture.Description.Height != height ||
        next_tex == nullptr)
    {
        if(next_tex)
        {
            m_Backend->unmapCudaSurface(&next_cuda_tex);
            m_Backend->destroyRenderResource(next_tex);
        }

        m_FrameTexture.Width = width;
        m_FrameTexture.Height = height;
        m_FrameTexture.Format = Tempest::DataFormat::RGBA8UNorm;

        auto& window = *m_Window;
        uint32_t window_width = window.getWidth();
        uint32_t window_height = window.getHeight();

        if(window_width != width ||
            window_height != height)
        {
            m_Flags |= DIRTY_RECT;
        }

        next_tex = m_Backend->createTexture(m_FrameTexture, Tempest::RESOURCE_DYNAMIC_DRAW);

        m_Backend->mapToCudaSurface(next_tex, cudaGraphicsRegisterFlagsSurfaceLoadStore, &next_cuda_tex);
    }

    m_RayTracer->draw(&next_cuda_tex, view_proj_inv);
}

void RayTracingCudaSystem::saveImage(const Path& file_path)
{
	updateBackbufferCopy();

	size_t dpitch = m_FrameTexture.Width*DataFormatElementSize(m_FrameTexture.Format);

	auto err = cudaMemcpy2DFromArray(m_BackbufferCopy->getData(), dpitch, m_CudaTexture.Array, 0, 0, dpitch, m_FrameTexture.Height, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy backbuffer texture to CPU visible memory");
		return;
	}

	SaveImage(m_BackbufferCopy->getHeader(), m_BackbufferCopy->getData(), file_path);
}

Texture* RayTracingCudaSystem::getLastFrameTexture()
{
	updateBackbufferCopy();

	size_t dpitch = m_FrameTexture.Width*DataFormatElementSize(m_FrameTexture.Format);

	auto err = cudaMemcpy2DFromArray(m_BackbufferCopy->getData(), dpitch, m_CudaTexture.Array, 0, 0, dpitch, m_FrameTexture.Height, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy backbuffer texture to CPU visible memory");
		return m_BackbufferCopy.get();
	}

    return m_BackbufferCopy.get();
}
}