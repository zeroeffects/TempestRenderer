#include "tempest/utils/testing.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/api-all.hh"

#define TEMPEST_RENDERING_SYSTEM Tempest::GLSystem

TGE_TEST("Testing the rendering context")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<TEMPEST_RENDERING_SYSTEM>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");
    
    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend);
    
    std::vector<Tempest::Vector2> arr{ Tempest::Vector2(0.0f, 0.0f),
                                       Tempest::Vector2(0.0f, 1.0f),
                                       Tempest::Vector2(1.0f, 0.0f),
                                       Tempest::Vector2(1.0f, 1.0f) };
    
    std::vector<Tempest::uint16> idx_arr{ 0, 1, 2, 3};
    
    auto vertex_buf = Tempest::CreateBuffer(&sys_obj->Backend, arr, Tempest::VBType::VertexBuffer);
    auto index_buf = Tempest::CreateBuffer(&sys_obj->Backend, idx_arr, Tempest::VBType::IndexBuffer);
    
    auto shader = Tempest::CreateShader(&sys_obj->ShaderCompiler, CURRENT_SOURCE_DIR "/test.tfx");
    TGE_ASSERT(shader, "Expecting successful compilation");
    auto res_table = shader->createResourceTable("GlobalsBuffer", 1);
    TGE_ASSERT(res_table, "Expecting valid resource table");
    Tempest::Matrix4 mat;
    mat.identity();
    mat.translate(Tempest::Vector2(-0.5f, -0.5f));
    res_table->setResource("Globals[0].Transform", mat);
    auto baked_table = Tempest::ExtractBakedResourceTable(res_table);
    
    TGE_ASSERT(shader, "Could not create shader file");
    
    std::vector<Tempest::VertexAttributeDescription> layout_arr
    {
        { 0, "VertexData", Tempest::DataFormat::RG32F, 0 }
    };
    
    auto input_layout = Tempest::CreateInputLayout(&sys_obj->Backend, shader.get(), layout_arr);
    
    auto* shader_ptr = shader.get();
    auto linked_shader_prog = shader_ptr->getUniqueLinkage(nullptr);
    
    typedef decltype(sys_obj->Backend) BackendType;
    BackendType::CommandBufferType::DrawBatchType batch;
    batch.PrimitiveType = Tempest::DrawModes::TriangleStrip;
    batch.VertexCount = static_cast<Tempest::uint32>(idx_arr.size());
    batch.ResourceTable = baked_table.get();
    batch.LinkedShaderProgram = linked_shader_prog;
    batch.IndexBuffer = index_buf.get();
    batch.InputLayout = input_layout.get();
    batch.VertexBuffers[0].VertexBuffer = vertex_buf.get();
    batch.VertexBuffers[0].Stride = sizeof(Tempest::Vector2);
    
    command_buf->enqueueBatch(batch);
    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    for(;;)
    {
        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}