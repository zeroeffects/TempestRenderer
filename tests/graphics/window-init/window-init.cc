#include "tempest/utils/testing.hh"
#include "tempest/graphics/opengl-backend/gl-all.hh"
#include "tempest/graphics/rendering-convenience.hh"

#define TEMPEST_RENDERING_SYSTEM Tempest::GLSystem

TGE_TEST("Testing window initialization")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<TEMPEST_RENDERING_SYSTEM>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");
    
    sys_obj->Window.show();
    
    for(;;)
    {
        sys_obj->Window.swapBuffers();
    }
}
