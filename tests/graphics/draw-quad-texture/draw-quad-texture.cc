#include "tempest/utils/testing.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/api-all.hh"

// This one includes comments because it is used as proof of concept.
#define TEMPEST_RENDERING_SYSTEM Tempest::GLSystem

TGE_TEST("Testing the rendering context")
{
    // Basically, you start by initializing a window. That's your usual engine init part.
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<TEMPEST_RENDERING_SYSTEM>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");
    
    // You are going to need a command buffer. This is just a persistent mapped buffer and
    // a CPU buffer that contains description of draw commands (batches).
    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;
    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);
    
    // Vertex buffer and index buffer.
    std::vector<Tempest::Vector2> arr{ Tempest::Vector2(0.0f, 0.0f),
                                       Tempest::Vector2(0.0f, 1.0f),
                                       Tempest::Vector2(1.0f, 0.0f),
                                       Tempest::Vector2(1.0f, 1.0f) };
    
    std::vector<Tempest::uint16> idx_arr{ 0, 1, 2, 3};
    
    auto vertex_buf = Tempest::CreateBuffer(&sys_obj->Backend, arr, Tempest::VBType::VertexBuffer);
    auto index_buf = Tempest::CreateBuffer(&sys_obj->Backend, idx_arr, Tempest::VBType::IndexBuffer);
    
    // Compile a shader. This project contains a complicated shader parser, but you can
    // probably get away with minimal shader preprocessor just to fill the gaps between
    // shader languages.
    auto shader = Tempest::CreateShader(&sys_obj->ShaderCompiler, CURRENT_SOURCE_DIR "/test.tfx");
    TGE_ASSERT(shader, "Expecting successful compilation");
    
    // A texture that we are going to attach to the draw batch's resource table.
    auto tex = Tempest::CreateTexture(&sys_obj->Backend, CURRENT_SOURCE_DIR "/Mandrill.tga");
    tex->setFilter(Tempest::FilterMode::Linear, Tempest::FilterMode::Linear, Tempest::FilterMode::Linear);
    tex->setWrapMode(Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp);
    
    // We are building a resource table per draw call. Bindless textures enable us to set them
    // directly without additional processing, which is cool. You can actually do the same thing without
    // bindless textures. Just keep a table of all texture slots and set them with a single function.
    auto var_table = shader->createResourceTable("Globals", 1);
    TGE_ASSERT(var_table, "Expecting valid resource table");
    Tempest::uint32 tex_id = 0;
    Tempest::Matrix4 mat;
    mat.identity();
    mat.translate(Tempest::Vector2(-0.5f, -0.5f));
    var_table->setResource("Globals.Transform", mat);
    auto baked_table = Tempest::ExtractBakedResourceTable(var_table);

    // Basically, that's what your texture manager is going to setup. In most cases array of texture
    auto res_table = shader->createResourceTable("Resources", 0);
    res_table->setResource("Texture", *tex);
    auto baked_res_table = Tempest::ExtractBakedResourceTable(res_table);

    TGE_ASSERT(shader, "Could not create shader file");
    
    Tempest::DataFormat rt_fmt = Tempest::DataFormat::RGBA8;

    auto state_obj = Tempest::CreateStateObject(&sys_obj->Backend, &rt_fmt, 1, Tempest::DataFormat::D24S8, shader.get(), Tempest::DrawModes::TriangleStrip);
    
    // That's the actual batch. It describes all the stuff required for sucessful draw call.
    // Also, you should pool allocate graphics device objects, so that you can get away with
    // 32-bit handles on 64-bit systems.
    typedef decltype(sys_obj->Backend) BackendType;
    BackendType::CommandBufferType::DrawBatchType batch;
    batch.VertexCount = static_cast<Tempest::uint16>(idx_arr.size());
    batch.ResourceTable = baked_table.get();
    batch.PipelineState = state_obj.get();
    batch.IndexBuffer = index_buf.get();
    batch.VertexBuffers[0].VertexBuffer = vertex_buf.get();
    batch.VertexBuffers[0].Stride = sizeof(Tempest::Vector2);
    
    // Baking the command buffer.
    
    // This is done on build thrad per batch.
    command_buf->enqueueBatch(batch);
    
    // This one is done after everything is done on the build thread.
    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    sys_obj->Backend.setActiveTextures(16);

    // And that's the render loop. We have prebaked everything, so we are not doing anything special.
    for(;;)
    {
        sys_obj->Backend.setTextures(baked_res_table.get());

        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}