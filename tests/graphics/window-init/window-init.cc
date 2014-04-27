#include "tempest/utils/testing.hh"
#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-window.hh"
#include "tempest/graphics/opengl-backend/gl-convenience.hh"

TGE_TEST("Testing window initialization")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateGLSystemAndWindow(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");
    
    sys_obj->Window.show();
    
    while(true)
        sys_obj->Window.swapBuffers();
}
