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
#include "tempest/utils/assert.hh"

#include <Windows.h>
#include <Xinput.h>

namespace Tempest
{
DWORD (WINAPI *_XInputGetCapabilities)(DWORD dwUserIndex, DWORD dwFlags, XINPUT_CAPABILITIES* pCapabilities);
DWORD (WINAPI *_XInputGetState)(DWORD dwUserIndex, XINPUT_STATE* pState);

ControllerLibrary::ControllerLibrary()
    :   m_XInputLibrary("xinput1_3.dll")
{
    if(m_XInputLibrary.loaded())
    {
        _XInputGetCapabilities = reinterpret_cast<decltype(_XInputGetCapabilities)>(m_XInputLibrary.getProcAddress("XInputGetCapabilities"));
        _XInputGetState = reinterpret_cast<decltype(_XInputGetState)>(m_XInputLibrary.getProcAddress("XInputGetState"));
    }
}

std::vector<ControllerDescription> ControllerLibrary::GetControllerDescriptions()
{
    std::vector<ControllerDescription> ctls;
    if(_XInputGetCapabilities == nullptr)
        return ctls;

    ControllerDescription desc;
    for(DWORD i = 0; i < XUSER_MAX_COUNT; ++i)
    {
        XINPUT_CAPABILITIES caps;
        if(_XInputGetCapabilities(i, 0, &caps) != ERROR_SUCCESS)
        {
            continue;
        }
        TGE_ASSERT(caps.Type == XINPUT_DEVTYPE_GAMEPAD, "Unexpected. XINPUT was not telling anything else at the time");
        switch(caps.SubType)
        {
        case XINPUT_DEVSUBTYPE_GAMEPAD: desc.Type = ControllerType::Gamepad; break;
        case XINPUT_DEVSUBTYPE_WHEEL: desc.Type = ControllerType::Wheel; break;
        case XINPUT_DEVSUBTYPE_ARCADE_STICK: desc.Type = ControllerType::ArcadeStick; break;
        case XINPUT_DEVSUBTYPE_FLIGHT_STICK: desc.Type = ControllerType::FlightStick; break;
        case XINPUT_DEVSUBTYPE_DANCE_PAD: desc.Type = ControllerType::DancePad; break;
        case XINPUT_DEVSUBTYPE_GUITAR: desc.Type = ControllerType::Guitar; break;
        case XINPUT_DEVSUBTYPE_GUITAR_ALTERNATE: desc.Type = ControllerType::GuitarAlternate; break;
        case XINPUT_DEVSUBTYPE_GUITAR_BASS: desc.Type = ControllerType::GuitarBass; break;
        case XINPUT_DEVSUBTYPE_DRUM_KIT: desc.Type = ControllerType::DrumKit; break;
        case XINPUT_DEVSUBTYPE_ARCADE_PAD: desc.Type = ControllerType::ArcadePad; break;
        default: TGE_ASSERT(false, "Unknown gamepad type");
        }
        desc.Identifier = i;
        desc.Name = "XINPUT Gamepad";
        if(caps.Flags & XINPUT_CAPS_FFB_SUPPORTED)
        {
            desc.Extra |= TEMPEST_CONTROLLER_FORCE_FEEDBACK;
        }
        if(caps.Flags & XINPUT_CAPS_WIRELESS)
        {
            desc.Extra |= TEMPEST_CONTROLLER_WIRELESS;
        }
        if(caps.Flags & XINPUT_CAPS_NO_NAVIGATION)
        {
            desc.Extra |= TEMPEST_CONTROLLER_NO_NAVIGATION;
        }
        if(caps.Flags & XINPUT_CAPS_VOICE_SUPPORTED)
        {
            desc.Extra |= TEMPEST_CONTROLLER_VOICE;
        }
        ctls.push_back(desc);
    }
    return ctls;
}

Controller::Controller(const ControllerDescription& desc)
    :   m_Index(static_cast<DWORD>(desc.Identifier))
{
}

Controller::~Controller()
{
}

Controller::Controller(Controller&& ctr)
{
    m_Index = ctr.m_Index;
}

Controller& Controller::operator=(Controller&& ctr)
{
    m_Index = ctr.m_Index;
    return *this;
}

static void SetButtonState(ControllerState* state, size_t button_mask, uint16 button_state)
{

}

bool Controller::getState(ControllerState* state)
{
    XINPUT_STATE xstate = {};
    if(_XInputGetState(m_Index, &xstate) != ERROR_SUCCESS)
    {
        return false;
    }

    auto compute_trigger = [](int16 state)
    {
        if(state == 255)
            return std::numeric_limits<int16>::max();
        else if(state == 0)
            return std::numeric_limits<int16>::min();
        return static_cast<int16>((state - 127) * 256);
    };
    state->LeftTrigger = compute_trigger(xstate.Gamepad.bLeftTrigger);
    state->RightTrigger = compute_trigger(xstate.Gamepad.bRightTrigger);
    state->LeftThumbX = abs(xstate.Gamepad.sThumbLX) > TEMPEST_GAMEPAD_LEFT_THUMB_DEADZONE ? xstate.Gamepad.sThumbLX : 0;
    state->LeftThumbY = abs(xstate.Gamepad.sThumbLY) > TEMPEST_GAMEPAD_LEFT_THUMB_DEADZONE ? xstate.Gamepad.sThumbLY : 0;
    state->RightThumbX = abs(xstate.Gamepad.sThumbRX) > TEMPEST_GAMEPAD_RIGHT_THUMB_DEADZONE ? xstate.Gamepad.sThumbRX : 0;
    state->RightThumbY = abs(xstate.Gamepad.sThumbRY) > TEMPEST_GAMEPAD_RIGHT_THUMB_DEADZONE ? xstate.Gamepad.sThumbRY : 0;

    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_UP)
        state->ButtonMask |= TEMPEST_CONTROLLER_DPAD_UP;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_DOWN)
        state->ButtonMask |= TEMPEST_CONTROLLER_DPAD_DOWN;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_LEFT)
        state->ButtonMask |= TEMPEST_CONTROLLER_DPAD_LEFT;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT)
        state->ButtonMask |= TEMPEST_CONTROLLER_DPAD_RIGHT;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_START)
        state->ButtonMask |= TEMPEST_CONTROLLER_START;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_BACK)
        state->ButtonMask |= TEMPEST_CONTROLLER_BACK;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_THUMB)
        state->ButtonMask |= TEMPEST_CONTROLLER_LEFT_THUMB;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_THUMB)
        state->ButtonMask |= TEMPEST_CONTROLLER_RIGHT_THUMB;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER)
        state->ButtonMask |= TEMPEST_CONTROLLER_LEFT_SHOULDER;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER)
        state->ButtonMask |= TEMPEST_CONTROLLER_RIGHT_SHOULDER;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_A)
        state->ButtonMask |= TEMPEST_CONTROLLER_A;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_B)
        state->ButtonMask |= TEMPEST_CONTROLLER_B;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_X)
        state->ButtonMask |= TEMPEST_CONTROLLER_X;
    if(xstate.Gamepad.wButtons & XINPUT_GAMEPAD_Y)
        state->ButtonMask |= TEMPEST_CONTROLLER_Y;
    return true;
}
}