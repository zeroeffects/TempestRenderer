#include "tempest/utils/testing.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/opengl-backend/gl-all.hh"

TGE_TEST("Testing the off-screen rendering context")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateGLSystemAndWindow(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");
    
    Tempest::GLRenderingBackend backend;
    auto command_buf = Tempest::CreateCommandBuffer(&backend);
    
    std::vector<Tempest::Vector2> arr{ Tempest::Vector2(0.0f, 0.0f),
                                       Tempest::Vector2(0.0f, 1.0f),
                                       Tempest::Vector2(1.0f, 0.0f),
                                       Tempest::Vector2(1.0f, 1.0f) };
    
    std::vector<Tempest::uint16> idx_arr{ 0, 1, 2, 3};
    
    auto vertex_buf = Tempest::CreateBuffer(&backend, arr, Tempest::VBType::VertexBuffer);
    auto index_buf = Tempest::CreateBuffer(&backend, idx_arr, Tempest::VBType::IndexBuffer);
    
    Tempest::GLShaderCompiler compiler;
    auto shader = Tempest::CreateShader(&compiler, CURRENT_SOURCE_DIR "/test.tfx");
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
        { 0, "VertexData", Tempest::DataFormat::RG32F, sizeof(Tempest::Vector2), 0}
    };
    
    auto input_layout = Tempest::CreateInputLayout(&backend, shader.get(), layout_arr);
    
    auto* shader_ptr = shader.get();
    auto linked_shader_prog = shader_ptr->getUniqueLinkage(nullptr);
    
    Tempest::GLDrawBatch batch;
    batch.PrimitiveType = Tempest::DrawModes::TriangleStrip;
    batch.VertexCount = idx_arr.size();
    batch.ResourceTable = baked_table.get();
    batch.LinkedShaderProgram = linked_shader_prog;
    batch.IndexBuffer = index_buf.get();
    batch.InputLayout = input_layout.get();
    batch.VertexBuffers[0] = vertex_buf.get();
    
    command_buf->enqueueBatch(batch);
    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    for(;;)
    {
        backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}