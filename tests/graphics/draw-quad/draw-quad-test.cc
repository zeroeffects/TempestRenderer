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

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;

    Tempest::DataFormat fmts[] =
    {
        Tempest::DataFormat::BGRA8UNorm
    };

    Tempest::ClearValue clear;
    clear.Color = {};

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
    auto res_table = shader->createResourceTable("Globals", 1);
    TGE_CHECK(res_table, "Expecting valid resource table");
    Tempest::Matrix4 mat;
    mat.identity();
	mat.translate(Tempest::Vector2{-0.5f, -0.5f});
    res_table->setResource("Globals[0].Transform", mat);
    auto baked_table = Tempest::ExtractBakedResourceTable(res_table);
    
    TGE_CHECK(shader, "Could not create shader file");
    
    Tempest::DataFormat rt_fmt = Tempest::DataFormat::RGBA8UNorm;
    
    auto pipeline_state = Tempest::CreateStateObject(&sys_obj->Backend, &rt_fmt, 1, Tempest::DataFormat::Unknown,
                                                     shader.get(), Tempest::DrawModes::TriangleStrip);
    
    typedef decltype(sys_obj->Backend) BackendType;
    BackendType::CommandBufferType::DrawBatchType batch;
    batch.VertexCount = static_cast<uint32_t>(idx_arr.size());
    batch.ResourceTable = baked_table.get();
    batch.IndexBuffer = index_buf.get();
    batch.PipelineState = pipeline_state.get();
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