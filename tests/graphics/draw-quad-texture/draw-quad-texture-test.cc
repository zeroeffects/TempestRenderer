#include "tempest/utils/testing.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/graphics/api-all.hh"

#define TEST_SYSTEM Tempest::PreferredSystem

TGE_TEST("Testing the rendering context")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<TEST_SYSTEM>(wdesc);
    TGE_CHECK(sys_obj, "GL initialization failed");
    
    Tempest::DataFormat fmts[] = { Tempest::DataFormat::BGRA8UNorm };

    Tempest::ClearValue clear;
    clear.Color = {};

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;
    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);
    
    std::vector<Tempest::Vector2> arr{ Tempest::Vector2{0.0f, 0.0f},
                                       Tempest::Vector2{0.0f, 1.0f},
                                       Tempest::Vector2{1.0f, 0.0f},
                                       Tempest::Vector2{1.0f, 1.0f} };
    
    std::vector<uint16_t> idx_arr{ 0, 1, 2, 3};
    
    auto vertex_buf = Tempest::CreateBuffer(&sys_obj->Backend, arr, Tempest:: ResourceBufferType::VertexBuffer);
    auto index_buf = Tempest::CreateBuffer(&sys_obj->Backend, idx_arr, Tempest:: ResourceBufferType::IndexBuffer);
    
    auto shader = Tempest::CreateShader(&sys_obj->ShaderCompiler, CURRENT_SOURCE_DIR "/test.tfx");
    TGE_CHECK(shader, "Expecting successful compilation");
    
    auto tex = Tempest::CreateTexture(&sys_obj->Backend, CURRENT_SOURCE_DIR "/Mandrill.tga");
    
    Tempest::SamplerDescription sampler_desc;
    sampler_desc.MagFilter = sampler_desc.MinFilter = sampler_desc.MipmapMode = Tempest::TextureSampling::Bilinear;

    auto var_table = shader->createResourceTable("Globals", 1);
    TGE_CHECK(var_table, "Expecting valid resource table");
    uint32_t tex_id = 0;
    Tempest::Matrix4 mat;
    mat.identity();
    mat.translate(Tempest::Vector2{-0.5f, -0.5f});
    var_table->setResource("Globals.Transform", mat);
    auto baked_table = Tempest::ExtractBakedResourceTable(var_table);

    auto res_table = shader->createResourceTable("Resources", 0);
    res_table->setResource("Texture", *tex);
    auto baked_res_table = Tempest::ExtractBakedResourceTable(res_table);

    TGE_CHECK(shader, "Could not create shader file");
    
    Tempest::DataFormat rt_fmt = Tempest::DataFormat::RGBA8;

    Tempest::BufferBinding buffer { 0, Tempest::InputRateType::PerVertex, sizeof(Tempest::Vector2) };

    auto state_obj = Tempest::CreateStateObject(&sys_obj->Backend, &rt_fmt, 1, Tempest::DataFormat::Unknown, shader.get(), Tempest::DrawModes::TriangleStrip);

    typedef decltype(sys_obj->Backend) BackendType;
    BackendType::CommandBufferType::DrawBatchType batch;
    batch.VertexCount = static_cast<uint16_t>(idx_arr.size());
    batch.ResourceTable = baked_table.get();
    batch.PipelineState = state_obj.get();
    batch.IndexBuffer = index_buf.get();
    batch.VertexBuffers[0].VertexBuffer = vertex_buf.get();
    
    command_buf->enqueueBatch(batch);

    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    while(!sys_obj->Window.isDead())
    {
        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}