#include "tempest/utils/testing.hh"
#include "tempest/graphics/preferred-backend.hh"
#include "tempest/graphics/rendering-convenience.hh"

TGE_TEST("Testing window initialization")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::PreferredSystem>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");
    
    sys_obj->Window.show();
    
    while(!sys_obj->Window.isDead())
    {
        sys_obj->Window.swapBuffers();
    }
}
