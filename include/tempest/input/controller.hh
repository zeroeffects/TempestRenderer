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

#include <cstdint>
#include <string>

#ifdef _WIN32
    #include "tempest/utils/library.hh"
#endif

#include <vector>

namespace Tempest
{
    
#define TEMPEST_GAMEPAD_LEFT_THUMB_DEADZONE   7849
#define TEMPEST_GAMEPAD_RIGHT_THUMB_DEADZONE  8689
#define TEMPEST_GAMEPAD_TRIGGER_THRESHOLD    -30000

#ifndef TEMPEST_CONTROLLER_EVENT_QUEUE_SIZE
    #define TEMPEST_CONTROLLER_EVENT_QUEUE_SIZE 256
#endif
    
enum class ControllerType
{
    Gamepad,
    Wheel,
    ArcadeStick,
    FlightStick,
    DancePad,
    Guitar,
    GuitarAlternate,
    GuitarBass,
    DrumKit,
    ArcadePad
};

enum ControllerExtra
{
    TEMPEST_CONTROLLER_FORCE_FEEDBACK = 1 << 0,
    TEMPEST_CONTROLLER_WIRELESS       = 1 << 1,
    TEMPEST_CONTROLLER_NO_NAVIGATION  = 1 << 2,
    TEMPEST_CONTROLLER_VOICE          = 1 << 3
};

struct ControllerDescription
{
    std::size_t          Identifier;
    std::string     Name;
    ControllerType  Type;
    size_t          Extra;
};

enum ControllerButtons
{
    TEMPEST_CONTROLLER_DPAD_UP        = 1 << 0,
    TEMPEST_CONTROLLER_DPAD_DOWN      = 1 << 1,
    TEMPEST_CONTROLLER_DPAD_LEFT      = 1 << 2,
    TEMPEST_CONTROLLER_DPAD_RIGHT     = 1 << 3,
    TEMPEST_CONTROLLER_START          = 1 << 4,
    TEMPEST_CONTROLLER_BACK           = 1 << 5,
    TEMPEST_CONTROLLER_LEFT_THUMB     = 1 << 6,
    TEMPEST_CONTROLLER_RIGHT_THUMB    = 1 << 7,
    TEMPEST_CONTROLLER_LEFT_SHOULDER  = 1 << 8,
    TEMPEST_CONTROLLER_RIGHT_SHOULDER = 1 << 9,
    TEMPEST_CONTROLLER_A              = 1 << 10,
    TEMPEST_CONTROLLER_B              = 1 << 11,
    TEMPEST_CONTROLLER_X              = 1 << 12,
    TEMPEST_CONTROLLER_Y              = 1 << 13
};

struct ControllerState
{
    uint16_t        ButtonMask   = 0;
    int16_t         LeftTrigger  = 0,
                    RightTrigger = 0,
                    LeftThumbX   = 0,
                    LeftThumbY   = 0,
                    RightThumbX  = 0,
                    RightThumbY  = 0;
};

class ControllerLibrary
{
#ifdef _WIN32
    Library         m_XInputLibrary;
#endif
public:
    ControllerLibrary();

    std::vector<ControllerDescription> GetControllerDescriptions();
};

class Controller
{
#ifdef LINUX
    int             m_FD;
#elif defined(_WIN32)
    DWORD           m_Index;
#else
#   error "Unsupported platform"
#endif
public:
    Controller(const ControllerDescription& desc);
     ~Controller();
     
    // It is not copyable because it is a connection to the device.
    Controller(const Controller&)=delete;
    Controller& operator=(const Controller&)=delete;
    
    Controller(Controller&&);
    Controller& operator=(Controller&&);
    
    bool getState(ControllerState* state);
};
}
