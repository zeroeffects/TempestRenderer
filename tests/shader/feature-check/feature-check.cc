#include "tempest/utils/testing.hh"
#include "tempest/shader/dx-shader-generator.hh"
#include "tempest/shader/gl-shader-generator.hh"

TGE_TEST("Testing shader compiler feature")
{
    {
        std::fstream fs("test.tfx", std::ios::out|std::ios::trunc);
            fs << R"(
                buffer TestCbuffer
                {
                    vec4 color;
                    vec3 normal;
                }
            )";
        fs.close();
        
        Tempest::Shader::EffectDescription fx;
        auto status = Tempest::DXFX::LoadEffect("test.tfx", nullptr, nullptr, 0, 0, fx);
        TGE_ASSERT(status, "Loading effect file failed.");
        
        status = Tempest::GLFX::LoadEffect("test.tfx", nullptr, nullptr, 0, 0, fx);
        TGE_ASSERT(status, "Loading effect file failed.");
    }
}