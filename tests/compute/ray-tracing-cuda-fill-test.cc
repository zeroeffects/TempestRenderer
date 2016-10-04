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

#include "tempest/utils/testing.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/compute/ray-tracing-cuda.hh"
#include "tempest/math/functions.hh"
#include "tempest/graphics/state-object.hh"
#include "tempest/utils/timer.hh"

#include "driver_types.h"

TGE_TEST("Testing volume ray tracing")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 640;
    wdesc.Height = 480;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::PreferredSystem>(wdesc);
    TGE_CHECK(sys_obj, "GL initialization failed");

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = wdesc.Width;
    tex_desc.Height = wdesc.Height;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;
    auto gpu_tex = Tempest::CreateTexture(&sys_obj->Backend, tex_desc, Tempest::RESOURCE_DYNAMIC_DRAW);

    auto cuda_tex = Tempest::MapCudaSurface(&sys_obj->Backend, gpu_tex, cudaGraphicsRegisterFlagsSurfaceLoadStore);

    std::unique_ptr<Tempest::RayTracerCuda> ray_trace_cuda(new Tempest::RayTracerCuda(&cuda_tex, Tempest::Matrix4::identityMatrix()));
    ray_trace_cuda->setBackgroundSpectrum(Tempest::SRGBToSpectrum(Tempest::Vector3{1.0f, 0.0f, 1.0f}));

    ray_trace_cuda->drawOnce();

    auto ray_trace_blit = Tempest::CreateShader(&sys_obj->ShaderCompiler, SOURCE_SHADING_DIR "/ray-trace-blit.tfx");
    TGE_CHECK(ray_trace_blit, "Failed to load ray trace backbuffer blit shader");

    auto rt_fmt = Tempest::DataFormat::RGBA8UNorm;
    Tempest::DepthStencilStates ds_state;
    ds_state.DepthTestEnable = true;
    ds_state.DepthWriteEnable = true;
    auto blit_state_obj = Tempest::CreateStateObject(&sys_obj->Backend, &rt_fmt, 1, Tempest::DataFormat::Unknown, ray_trace_blit.get(), Tempest::DrawModes::TriangleList, nullptr, nullptr, &ds_state);

    Tempest::PreferredBackend::CommandBufferType::DrawBatchType blit_batch;
    blit_batch.VertexCount = 3;
    blit_batch.PipelineState = blit_state_obj.get();
    blit_batch.ResourceTable = nullptr;

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = 1;
    cmd_buffer_desc.ConstantsBufferSize = 1024;
    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);
    command_buf->enqueueBatch(blit_batch);
    command_buf->prepareCommandBuffer();

    sys_obj->Window.show();

    uint32_t tex_table_slots = 5;
    uint32_t aux_slots = 1;
    uint32_t total_slots = tex_table_slots + aux_slots;
    Tempest::BakedResourceTable textures(total_slots*4*sizeof(float));

    memset(textures.get(), 0, total_slots*4*sizeof(float));
    *reinterpret_cast<uint64_t*>(textures.get() + tex_table_slots*4*sizeof(float)) = gpu_tex->getHandle();

    sys_obj->Backend.setActiveTextures(total_slots);

    sys_obj->Backend.setViewportRect(0, 0, wdesc.Width, wdesc.Height);

    while(!sys_obj->Window.isDead())
    {
        sys_obj->Backend.clearColorBuffer(0, Tempest::Vector4{1.0f, 0.0f, 0.0f, 0.0f});
        sys_obj->Backend.clearDepthStencilBuffer();

        sys_obj->Backend.setTextures(&textures);

        sys_obj->Backend.submitCommandBuffer(command_buf.get());

        sys_obj->Window.swapBuffers();
    }
}
