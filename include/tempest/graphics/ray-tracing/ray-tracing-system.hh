/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 Zdravko Velinov
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

#ifndef _TEMPEST_RAY_TRACING_SYSTEM_HH_
#define _TEMPEST_RAY_TRACING_SYSTEM_HH_

#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/image/image.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/debug/fps-counter.hh"

namespace Tempest
{
// This basically wraps both the ray tracer and the windowing functionality
class RayTracingSystem
{
    typedef Tempest::PreferredBackend::IOCommandBufferType IOCommandBufferType;
    typedef Tempest::PreferredBackend::CommandBufferType CommandBufferType;
    typedef Tempest::PreferredBackend::StorageType StorageType;
    typedef Tempest::PreferredBackend::TextureType TextureType;
    typedef Tempest::PreferredBackend::StateObjectType StateObjectType;
    typedef Tempest::PreferredShaderCompiler::ShaderProgramType ShaderType;

    const FrameData*                    m_LastFrame = nullptr;
    Tempest::TextureDescription         m_FrameTexture;
    std::unique_ptr<PreferredSystem>    m_System;
    CommandBufferType*                  m_CommandBuffer = nullptr;
    IOCommandBufferType*                m_IOCommandBuffer = nullptr;
    StorageType*                        m_Storage = nullptr;
    TextureType*                        m_GPUTexture = nullptr;
    StateObjectType*                    m_StateObject = nullptr;
    ShaderType*                         m_BlitShader = nullptr;
    std::unique_ptr<FpsCounter>         m_FpsCounter;
    PreferredBackend::CommandBufferType* m_FpsCommandBuffer;
    bool								m_EnableFpsCounter = false;

	uint32_t                            m_Flags = DIRTY_RECT;

    std::unique_ptr<RayTracerScene>     m_RayTracer;

    Tempest::BakedResourceTable         m_TextureBinding;

public:
    static const uint32_t TextureTableSlots = 5;
    static const uint32_t AuxTextureSlots = 1;

    RayTracingSystem(uint32_t image_width, uint32_t image_height, const Matrix4& view_proj_inv, const RTSettings& settings = RTSettings())
        :   m_RayTracer(new Tempest::RayTracerScene(image_width, image_height, view_proj_inv)),
            m_TextureBinding((TextureTableSlots + AuxTextureSlots)*4*sizeof(float))
    {
        memset(m_TextureBinding.get(), 0, (TextureTableSlots + AuxTextureSlots)*4*sizeof(float));
		m_FrameTexture.Width = image_width;
        m_FrameTexture.Height = image_height;
        m_FrameTexture.Format = Tempest::DataFormat::RGBA8UNorm;
    }

    ~RayTracingSystem()
    {
        if(m_System)
        {
            auto& backend = m_System->Backend;
            m_System->ShaderCompiler.destroyRenderResource(m_BlitShader);
            backend.destroyRenderResource(m_StateObject);
            backend.destroyRenderResource(m_CommandBuffer);
            backend.destroyRenderResource(m_IOCommandBuffer);
            backend.destroyRenderResource(m_Storage);
            backend.destroyRenderResource(m_GPUTexture);
        }
    }

    Tempest::RayTracerScene* getRayTracer() { return m_RayTracer.get(); }

    void startRendering()
    {
        m_RayTracer->initWorkers();
    }
    
    void completeFrame()
    {
        auto frame_data = m_LastFrame = m_RayTracer->drawOnce();
        auto& hdr = frame_data->Backbuffer->getHeader();
        m_FrameTexture.Width = hdr.Width;
        m_FrameTexture.Height = hdr.Height;
        m_FrameTexture.Format = Tempest::DataFormat::RGBA8UNorm;
		if(m_FrameTexture.Width != hdr.Width &&
		   m_FrameTexture.Height != hdr.Height)
		    m_Flags |= DIRTY_RECT;
    }

    void completeFrameAndRestart(uint32_t width, uint32_t height, const Matrix4& view_proj_inv)
    {
        auto frame_data = m_LastFrame = m_RayTracer->draw(width, height, view_proj_inv);
        auto& hdr = frame_data->Backbuffer->getHeader();
        m_FrameTexture.Width = hdr.Width;
        m_FrameTexture.Height = hdr.Height;
        m_FrameTexture.Format = Tempest::DataFormat::RGBA8UNorm;
		if(m_FrameTexture.Width != hdr.Width &&
		   m_FrameTexture.Height != hdr.Height)
		    m_Flags |= DIRTY_RECT;
    }

    bool saveImage(const Path& file_path)
    {
		TGE_ASSERT(m_LastFrame, "You must perform at least one draw before saving image");
        auto* backbuffer = m_LastFrame->Backbuffer.get();
        return SaveImage(backbuffer->getHeader(), backbuffer->getData(), file_path);
    }

    void displayImage()
    {
		TGE_ASSERT(m_LastFrame, "You must perform at least one draw before displaying data");
        auto* backbuffer = m_LastFrame->Backbuffer.get();
        DisplayImage(backbuffer->getHeader(), backbuffer->getData());
    }

    PreferredWindow& getWindow()
    {
        updateSystem();
        return m_System->Window;
    }

    PreferredBackend& getBackend()
    {
        return m_System->Backend;
    }

    PreferredShaderCompiler& getShaderCompiler()
    {
        return m_System->ShaderCompiler;
    }

    const Texture* getLastFrameTexture() const
    {
        return m_LastFrame->Backbuffer.get();
    }

	bool submitFrame()
	{
		TGE_ASSERT(m_LastFrame, "You must draw before presenting an image");
        
        updateSystem();

        if(m_System->Window.isDead())
            return false;

        auto& backend = m_System->Backend;
        auto& window = m_System->Window;
        if((m_Flags & DIRTY_RECT) ||
		   !m_IOCommandBuffer)
        {
            // TODO: fix infinite realloc
            if(m_IOCommandBuffer)
            {
                backend.destroyRenderResource(m_IOCommandBuffer);
                backend.destroyRenderResource(m_Storage);
                backend.destroyRenderResource(m_GPUTexture);
            }

	        Tempest::IOCommandBufferDescription io_cmd_buf_desc;
            io_cmd_buf_desc.CommandCount = 1;
            m_IOCommandBuffer = backend.createIOCommandBuffer(io_cmd_buf_desc);
            m_Storage = backend.createStorageBuffer(Tempest::StorageMode::PixelUnpack, m_FrameTexture.Width*m_FrameTexture.Height*sizeof(uint32_t));

	        m_GPUTexture = backend.createTexture(m_FrameTexture, Tempest::RESOURCE_DYNAMIC_DRAW);

            Tempest::PreferredBackend::IOCommandBufferType::IOCommandType io_command;
            io_command.CommandType = Tempest::IOCommandMode::CopyStorageToTexture;
            io_command.Source.Storage = m_Storage;
            io_command.Destination.Texture = m_GPUTexture;
            io_command.SourceOffset = 0;
            io_command.DestinationCoordinate.X = io_command.DestinationCoordinate.Y = 0;
            io_command.Width = m_FrameTexture.Width;
            io_command.Height = m_FrameTexture.Height;
            m_IOCommandBuffer->enqueueCommand(io_command);

            *reinterpret_cast<uint64_t*>(m_TextureBinding.get() + TextureTableSlots*4*sizeof(float)) = m_GPUTexture->getHandle();

            window.resize(m_FrameTexture.Width, m_FrameTexture.Height);

			m_Flags &= ~DIRTY_RECT;
        }

        auto* backbuffer = m_LastFrame->Backbuffer.get();
        auto& hdr = backbuffer->getHeader();
        m_Storage->storeTexture(0, backbuffer->getHeader(), backbuffer->getData());

        backend.setViewportRect(0, 0, hdr.Width, hdr.Height);

        backend.submitCommandBuffer(m_IOCommandBuffer);

        backend.clearColorBuffer(0, Tempest::Vector4{1.0f, 0.0f, 0.0f, 0.0f});
        backend.clearDepthStencilBuffer();

        backend.setTextures(&m_TextureBinding);

        backend.submitCommandBuffer(m_CommandBuffer);

        if(m_FpsCounter->update(m_FrameTexture.Width, m_FrameTexture.Height))
        {
            m_FpsCommandBuffer->clear();
            auto draw_batches = m_FpsCounter->getDrawBatches();
            auto batch_count = m_FpsCounter->getDrawBatchCount();
            for(decltype(batch_count) i = 0; i < batch_count; ++i)
            {
                m_FpsCommandBuffer->enqueueBatch(draw_batches[i]);
            }

            m_FpsCommandBuffer->prepareCommandBuffer();
        }

		if(m_EnableFpsCounter)
		{
			backend.submitCommandBuffer(m_FpsCommandBuffer);
		}

		return true;
	}

    bool presentFrame()
    {
        if(!submitFrame())
			return false;
        
        m_System->Window.swapBuffers();
        return true;
    }

    void setEnableFpsCounter(bool fps_counter) { m_EnableFpsCounter = fps_counter; }
    void toggleFpsCounter() { m_EnableFpsCounter = !m_EnableFpsCounter; }
    bool getFpsCounterStatus() { return m_EnableFpsCounter; }

private:
    void updateSystem()
    {
        if(!m_System)
        {
            WindowDescription wdesc;
            wdesc.Width = m_FrameTexture.Width;
            wdesc.Height = m_FrameTexture.Height;
            wdesc.Title = "Test window";
            m_System = Tempest::CreateSystemAndWindowSimple<Tempest::PreferredSystem>(wdesc);
            TGE_ASSERT(m_System, "GL initialization failed");

            uint32_t total_slots = TextureTableSlots + AuxTextureSlots;
            m_System->Backend.setActiveTextures(total_slots);

            BasicFileLoader loader;
            m_BlitShader = m_System->ShaderCompiler.compileShaderProgram(SOURCE_SHADING_DIR "/ray-trace-blit.tfx", &loader);
            TGE_ASSERT(m_BlitShader, "Failed to load ray trace backbuffer blit shader");

            auto rt_fmt = DataFormat::RGBA8UNorm;
            DepthStencilStates ds_state;
            ds_state.DepthTestEnable = true;
            ds_state.DepthWriteEnable = true;
            m_StateObject = m_System->Backend.createStateObject(&rt_fmt, 1, Tempest::DataFormat::Unknown, m_BlitShader, Tempest::DrawModes::TriangleList, nullptr, nullptr, &ds_state);

            CommandBufferType::DrawBatchType blit_batch;
            blit_batch.VertexCount = 3;
            blit_batch.PipelineState = m_StateObject;
            blit_batch.ResourceTable = nullptr;

            CommandBufferDescription cmd_buffer_desc;
            cmd_buffer_desc.CommandCount = 1;
            cmd_buffer_desc.ConstantsBufferSize = 1024;
            m_CommandBuffer = m_System->Backend.createCommandBuffer(cmd_buffer_desc);
            m_CommandBuffer->enqueueBatch(blit_batch);
            m_CommandBuffer->prepareCommandBuffer();

            Tempest::SubdirectoryFileLoader subdir_loader(SOURCE_SHADING_DIR);
            m_FpsCounter = std::unique_ptr<FpsCounter>(new FpsCounter(&m_System->Backend, &m_System->ShaderCompiler, &subdir_loader, 50.0f, static_cast<float>(m_FrameTexture.Width), static_cast<float>(m_FrameTexture.Height)));

            cmd_buffer_desc.CommandCount = 16;
            cmd_buffer_desc.ConstantsBufferSize = 1024;
            m_FpsCommandBuffer = m_System->Backend.createCommandBuffer(cmd_buffer_desc);

            m_System->Window.show();
        }
    }
};
}

#endif // _TEMPEST_RAY_TRACING_SYSTEM_HH_