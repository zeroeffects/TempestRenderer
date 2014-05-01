/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2014 Zdravko Velinov
 *   
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *   THE SOFTWARE.
 */

#include "tempest/input/controller.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/utils/logging.hh"

#include <linux/joystick.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include <limits>

#define TEMPEST_GAMEPAD_LEFT_THUMB_DEADZONE   7849
#define TEMPEST_GAMEPAD_RIGHT_THUMB_DEADZONE  8689
#define TEMPEST_GAMEPAD_TRIGGER_THRESHOLD    -30000

namespace Tempest
{
std::vector<ControllerDescription> GetControllerDescriptions()
{
    std::vector<ControllerDescription> desc_arr;
    
    Directory dir(Path("/dev/input"));
    for(auto& file : dir)
    {
        auto filename = file.filename();
        if(filename.compare(0, 2, "js") != 0)
            continue;
        
        std::stringstream num_ss(filename.substr(2));

        int fd = open("/dev/input/js0", O_RDONLY);        
        auto scope_obj = CreateAtScopeExit([fd]() { if(fd != -1) close(fd); });
        if(fd == -1)
            continue;
        
        #define JS_NAME_SIZE__ 256
        char name[JS_NAME_SIZE__];
        ioctl(fd, JSIOCGNAME(JS_NAME_SIZE__), name);
        
        ControllerDescription desc;
        desc.Name = name;
        num_ss >> desc.Identifier;
        if(!num_ss)
            continue;
        desc.Extra = 0;
        
        int ffb_n = 0;
        ioctl(fd, EVIOCGEFFECTS, &ffb_n);
        if(ffb_n != 0)
            desc.Extra |= TEMPEST_CONTROLLER_FORCE_FEEDBACK;
        
        desc.Type = ControllerType::Gamepad;
        
        desc_arr.push_back(desc);
    }

    return desc_arr;
}
    
Controller::Controller(const ControllerDescription& desc)
    :   m_FD(-1)
{
    std::stringstream ss;
    ss << "/dev/input/js" << desc.Identifier;
    m_FD = open(ss.str().c_str(), O_RDONLY|O_NONBLOCK);
    if(m_FD == -1)
       Log(LogLevel::Error, "Failed to open joystick device: ", ss); 
}

Controller::~Controller()
{
    if(m_FD != -1)
        ::close(m_FD);
}

Controller::Controller(Controller&& ctr)
    :   m_FD(ctr.m_FD)
{
    ctr.m_FD = -1;
}
    
Controller& Controller::operator=(Controller&& ctr)
{
    m_FD = ctr.m_FD;
    ctr.m_FD = -1;
}

static void SetButtonState(ControllerState* state, size_t button_mask, uint16 button_state)
{
    if(button_state)
        state->ButtonMask |= button_mask;
    else
        state->ButtonMask &= ~button_mask;
}

bool Controller::getState(ControllerState* state)
{
    if(m_FD == -1)
        return false;
    
    TGE_ASSERT(state, "Valid state must be passed to this function");
    
    constexpr int event_buffer_size = 256;
    
    struct js_event evt;
    
    int evt_last = 0;
    bool ret = false;
    while(1)
    {
        int read_bytes = read(m_FD, &evt, sizeof(js_event));
        if(read_bytes != sizeof(js_event))
            return ret;
        
        ret = true;
        
        int evt_last = read_bytes / sizeof(js_event);
        
        switch(evt.type & ~JS_EVENT_INIT)
        {
        case JS_EVENT_BUTTON:
        {
            switch(evt.number)
            {
            case 0: SetButtonState(state, TEMPEST_CONTROLLER_A, evt.value); break;
            case 1: SetButtonState(state, TEMPEST_CONTROLLER_B, evt.value); break;
            case 2: SetButtonState(state, TEMPEST_CONTROLLER_X, evt.value); break;
            case 3: SetButtonState(state, TEMPEST_CONTROLLER_Y, evt.value); break;
            case 4: SetButtonState(state, TEMPEST_CONTROLLER_LEFT_SHOULDER, evt.value); break;
            case 5: SetButtonState(state, TEMPEST_CONTROLLER_RIGHT_SHOULDER, evt.value); break;
            case 6: SetButtonState(state, TEMPEST_CONTROLLER_BACK, evt.value); break;
            case 7: SetButtonState(state, TEMPEST_CONTROLLER_START, evt.value); break;
            case 9: SetButtonState(state, TEMPEST_CONTROLLER_LEFT_THUMB, evt.value); break;
            case 10: SetButtonState(state, TEMPEST_CONTROLLER_RIGHT_THUMB, evt.value); break;
            
            case 8: // unsupported
            default: break;
            }
        } break;
        case JS_EVENT_AXIS:
        {
            switch(evt.number)
            {
            case 0: state->LeftThumbX = abs(evt.value) < TEMPEST_GAMEPAD_LEFT_THUMB_DEADZONE ? 0 : evt.value; break;
            case 1: state->LeftThumbY = abs(evt.value) < TEMPEST_GAMEPAD_LEFT_THUMB_DEADZONE ? 0 : evt.value; break;
            case 2: state->RightThumbX = abs(evt.value) < TEMPEST_GAMEPAD_RIGHT_THUMB_DEADZONE ? 0 : evt.value; break;
            case 3: state->RightThumbY = abs(evt.value) < TEMPEST_GAMEPAD_RIGHT_THUMB_DEADZONE ? 0 : evt.value; break;
            case 4: state->RightTrigger = evt.value < TEMPEST_GAMEPAD_TRIGGER_THRESHOLD ? std::numeric_limits<int16>::min() : evt.value; break;
            case 5: state->LeftTrigger = evt.value < TEMPEST_GAMEPAD_TRIGGER_THRESHOLD ? std::numeric_limits<int16>::min() : evt.value; break;
            case 6:
            {
                if(evt.value == 0)
                    state->ButtonMask &= ~(TEMPEST_CONTROLLER_DPAD_LEFT|TEMPEST_CONTROLLER_DPAD_RIGHT);
                else
                    state->ButtonMask |= evt.value < 0 ? TEMPEST_CONTROLLER_DPAD_LEFT : TEMPEST_CONTROLLER_DPAD_RIGHT;
            } break;
            case 7:
            {
                if(evt.value == 0)
                    state->ButtonMask &= ~(TEMPEST_CONTROLLER_DPAD_UP|TEMPEST_CONTROLLER_DPAD_DOWN);
                else
                    state->ButtonMask |= evt.value < 0 ? TEMPEST_CONTROLLER_DPAD_UP : TEMPEST_CONTROLLER_DPAD_DOWN;
            } break;
            }
        } break;
        default:
            Log(LogLevel::Warning, "Unknown controller event type: ", evt.type);
        }
    }
    
    return ret;
}
}