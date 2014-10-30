#include "tempest/utils/testing.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/opengl-backend/gl-all.hh"
#include "tempest/mesh/obj-loader.hh"

TGE_TEST("Testing loading object files directly into the engine for testing purposes")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::GLSystem>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");
    
    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend);
    
    auto shader_wo_tex = Tempest::CreateShader(&sys_obj->ShaderCompiler, SOURCE_SHADING_DIR "/blinn-phong.tfx", "PN");
    auto shader_tex = Tempest::CreateShader(&sys_obj->ShaderCompiler, SOURCE_SHADING_DIR "/blinn-phong.tfx", "PTN");

    typedef decltype(sys_obj->Backend) BackendType;
    BackendType::ShaderProgramType* shaders[] = { shader_tex.get(), shader_wo_tex.get() };
    
    auto batches = CREATE_SCOPED(BackendType::CommandBufferType::DrawBatchType*, [](BackendType::CommandBufferType::DrawBatchType* ptr) { delete[] ptr; });
    size_t batch_count = 0;
    auto ret = Tempest::LoadObjFileStaticGeometry(TEST_ASSETS_DIR "/cube/cube.obj", nullptr, shaders, &sys_obj->Backend, &batch_count, &batches);
    TGE_ASSERT(ret, "Failed to load test assets");
    
    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    for(;;)
    {
        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}