#include "tempest/utils/testing.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/utils/timer.hh"

#define TEST_SYSTEM Tempest::PreferredSystem

TGE_TEST("Testing window initialization")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<TEST_SYSTEM>(wdesc);
    TGE_CHECK(sys_obj, "GL initialization failed");
    
    sys_obj->Window.show();
    
    Tempest::TimeQuery timer;

    uint64_t period = 1000000ULL;

    while(!sys_obj->Window.isDead())
    {
        Tempest::Vector4 clear_color = {};
        clear_color.w = 1.0f;
        Array(clear_color)[(timer.time()/period) % 3] = 1.0f;
        
        sys_obj->Backend.clearColorBuffer(0, clear_color);

        sys_obj->Window.swapBuffers();
    }
}
