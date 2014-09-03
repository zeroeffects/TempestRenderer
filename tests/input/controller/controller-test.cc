#include "tempest/utils/testing.hh"
#include "tempest/input/controller.hh"

#include <chrono>
#include <thread>

TGE_TEST("Testing controller capabilities")
{
    std::vector<Tempest::ControllerDescription> controllers_descs = Tempest::GetControllerDescriptions();
    
    std::vector<Tempest::Controller> controllers;
      
    for(auto& cont_desc : controllers_descs)
    {
        controllers.push_back(Tempest::Controller(cont_desc));
    }

    Tempest::ControllerState state;
    
    for(;;)
    {
        for(auto& cont : controllers)
        {
            cont.getState(&state);
            if(state.ButtonMask || state.LeftThumbX || state.LeftThumbY ||
               state.RightThumbX || state.RightThumbY || state.LeftTrigger > -30000 || state.RightTrigger > -30000)
            {
                Tempest::Log(Tempest::LogLevel::Info, "X1: ", state.LeftThumbX, " Y1: ", state.LeftThumbY,
                                                     " X2: ", state.RightThumbX, " Y2: ", state.RightThumbY,
                                                     " DU: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_DPAD_UP) != 0,
                                                     " DD: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_DPAD_DOWN) != 0,
                                                     " DL: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_DPAD_LEFT) != 0,
                                                     " DR: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_DPAD_RIGHT) != 0,
                                                     " A: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_A) != 0,
                                                     " B: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_B) != 0,
                                                     " X: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_X) != 0,
                                                     " Y: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_Y) != 0,
                                                     " LS: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_LEFT_SHOULDER) != 0,
                                                     " RS: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_RIGHT_SHOULDER) != 0,
                                                     " LT: ", state.LeftTrigger,
                                                     " RT: ", state.RightTrigger,
                                                     " Start: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_START) != 0,
                                                     " Back: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_BACK) != 0,
                                                     " TL: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_LEFT_THUMB) != 0,
                                                     " TR: ", (state.ButtonMask & Tempest::TEMPEST_CONTROLLER_RIGHT_THUMB) != 0
                            );
            }
            std::chrono::milliseconds dura(20);
            std::this_thread::sleep_for(dura);
        }
    }
}