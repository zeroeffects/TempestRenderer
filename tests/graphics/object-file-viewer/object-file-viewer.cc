#include "tempest/utils/testing.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/mesh/obj-loader.hh"

TGE_TEST("Testing loading object files directly into the engine for testing purposes")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::GLSystem>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;

    typedef decltype(sys_obj->Backend) BackendType;
    typedef BackendType::ShaderProgramType ShaderProgramType;
    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);
    
    const size_t base_layout = 2;
    const size_t base_models = 3;
    Tempest::string models[base_models - 1 + base_layout - 1] = { "MaterialTexture", "MaterialAmbient", "MaterialSpecular" };
    std::vector<Tempest::UniqueResource<decltype(sys_obj->ShaderCompiler), ShaderProgramType>> shader_table;
    shader_table.reserve(base_layout*base_models);
    ShaderProgramType* shader_ptr_table[base_layout*base_models];

    for(Tempest::uint32 i = 0; i < base_layout; ++i)
    {
        for(Tempest::uint32 j = 0; j < base_models; ++j)
        {
            Tempest::uint32 idx = i * base_models + j;
            auto shader = Tempest::CreateShader(&sys_obj->ShaderCompiler, SOURCE_SHADING_DIR "/blinn-phong.tfx", models + 1 - i, i + j);
            shader_ptr_table[idx] = shader.get();
            shader_table.push_back(std::move(shader));
        }
    }

    auto mesh_blob = Tempest::LoadObjFileStaticGeometryBlob(TEST_ASSETS_DIR "/cube/cube.obj", nullptr, shader_ptr_table, &sys_obj->Backend);
    TGE_ASSERT(mesh_blob, "Failed to load test assets");
    
    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    for(;;)
    {
        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}