#include "tempest/utils/testing.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/opengl-backend/gl-all.hh"

// This one includes comments because it is used as proof of concept.

TGE_TEST("Testing the rendering context")
{
    // Basically, you start by initializing a window. That's your usual engine init part.
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateGLSystemAndWindow(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");
    
    // You are going to need a command buffer. This is just a persistent mapped buffer and
    // a CPU buffer that contains description of draw commands (batches).
    Tempest::GLRenderingBackend backend;
    auto command_buf = Tempest::CreateCommandBuffer(&backend);
    
    // Vertex buffer and index buffer.
    std::vector<Tempest::Vector2> arr{ Tempest::Vector2(0.0f, 0.0f),
                                       Tempest::Vector2(0.0f, 1.0f),
                                       Tempest::Vector2(1.0f, 0.0f),
                                       Tempest::Vector2(1.0f, 1.0f) };
    
    std::vector<Tempest::uint16> idx_arr{ 0, 1, 2, 3};
    
    auto vertex_buf = Tempest::CreateBuffer(&backend, arr, Tempest::VBType::VertexBuffer);
    auto index_buf = Tempest::CreateBuffer(&backend, idx_arr, Tempest::VBType::IndexBuffer);
    
    // Compile a shader. This project contains a complicated shader parser, but you can
    // probably get away with minimal shader preprocessor just to fill the gaps between
    // shader languages.
    Tempest::GLShaderCompiler compiler;
    auto shader = Tempest::CreateShader(&compiler, CURRENT_SOURCE_DIR "/test.tfx");
    TGE_ASSERT(shader, "Expecting successful compilation");
    
    // A texture that we are going to attach to the draw batch's resource table.
    auto tex = Tempest::CreateTexture(&backend, CURRENT_SOURCE_DIR "/Mandrill.tga");
    tex->setFilter(Tempest::FilterMode::Linear, Tempest::FilterMode::Linear, Tempest::FilterMode::Linear);
    tex->setWrapMode(Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp);
    
    // We are building a resource table per draw call. Bindless textures enable us to set them
    // directly without additional processing, which is cool. You can actually do the same thing without
    // bindless textures. Just keep a table of all texture slots and set them with a single function.
    auto res_table = shader->createResourceTable("GlobalsBuffer", 1);
    TGE_ASSERT(res_table, "Expecting valid resource table");
    Tempest::Matrix4 mat;
    mat.identity();
    mat.translate(Tempest::Vector2(-0.5f, -0.5f));
    res_table->setResource("Globals.Transform", mat);
    res_table->setResource("Globals.Texture", *tex);
    auto baked_table = Tempest::ExtractBakedResourceTable(res_table);
    
    TGE_ASSERT(shader, "Could not create shader file");
    
    // Vertex format. Nothing special about it.
    std::vector<Tempest::VertexAttributeDescription> layout_arr
    {
        { 0, "VertexData", Tempest::DataFormat::RG32F, 0 }
    };
    
    auto input_layout = Tempest::CreateInputLayout(&backend, shader.get(), layout_arr);
    
    auto* shader_ptr = shader.get();
    
    // Unique linkage because we might want to use subroutines for some reason.
    auto linked_shader_prog = shader_ptr->getUniqueLinkage(nullptr);
    
    // That's the actual batch. It describes all the stuff required for sucessful draw call. You
    // might want to attach a pipeline state to your draw batch. Also, you should pool allocate graphics
    // device objects, so that you can get away with 32-bit handles on 64-bit systems.
    Tempest::GLDrawBatch batch;
    batch.PrimitiveType = Tempest::DrawModes::TriangleStrip;
    batch.VertexCount = idx_arr.size();
    batch.ResourceTable = baked_table.get();
    batch.LinkedShaderProgram = linked_shader_prog;
    batch.IndexBuffer = index_buf.get();
    batch.InputLayout = input_layout.get();
    batch.VertexBuffers[0].VertexBuffer = vertex_buf.get();
    batch.VertexBuffers[0].Stride = sizeof(Tempest::Vector2);
    
    // Baking the command buffer.
    
    // This is done on build thrad per batch.
    command_buf->enqueueBatch(batch);
    
    // This one is done after everything is done on the build thread.
    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    // And that's the render loop. We have prebaked everything, so we are not doing anything special.
    for(;;)
    {
        backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}