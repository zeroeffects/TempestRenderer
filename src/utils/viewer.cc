/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2016 Zdravko Velinov
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

#include "tempest/utils/viewer.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/compute/ray-tracing-cuda-system.hh"

namespace Tempest
{
template<class TRTSystem>
RayTracingView<TRTSystem>::RayTracingView(TRTSystem& rt_sys, uint32_t image_width, uint32_t image_height, FreeCamera& cam)
    :   m_RTSystem(rt_sys),
        m_Camera(cam),
        m_Width(image_width),
        m_Height(image_height),
        m_ViewProjectionInverse(ComputeViewProjectionInverse(cam))
{
    auto& window = rt_sys.getWindow();
    window.setEventMask(Tempest::COLLECT_MOUSE_EVENTS|Tempest::COLLECT_WINDOW_EVENTS);

    m_MousePosition = { (float)window.getMouseX(), (float)window.getMouseY() };
    m_PreviousMousePosition = m_MousePosition;

    m_PreviousTime = m_Timer.time();
}

template<class TRTSystem>
void RayTracingView<TRTSystem>::handleEvent(Tempest::WindowSystemEvent wevent)
{
    switch(wevent.Type)
    {
    case Tempest::WindowEventType::MouseButtonPressed:
    {
        if(!m_Active)
            break;

        switch(wevent.MouseButton)
        {
        case Tempest::MouseButtonId::RightButton:
        {
            m_Movement |= MOVEMENT_ROTATION;
        } break;
		case Tempest::MouseButtonId::MiddleButton:
		{
			m_Movement |= MOVEMENT_DRAG;
		} break;
        }
    } break;
    case Tempest::WindowEventType::MouseButtonReleased:
    {
        switch(wevent.MouseButton)
        {
        case Tempest::MouseButtonId::RightButton:
        {
            m_Movement &= ~MOVEMENT_ROTATION;
        } break;
		case Tempest::MouseButtonId::MiddleButton:
		{
			m_Movement &= ~MOVEMENT_DRAG;
		} break;
        }
    } break;
    case Tempest::WindowEventType::MouseMoved:
    {
        m_MousePosition = Tempest::Vector2{ (float)wevent.MouseMoved.MouseX, (float)wevent.MouseMoved.MouseY };
    } break;
    case Tempest::WindowEventType::Focus:
    {
        m_Active = wevent.Enabled != 0;
		m_Movement = 0;
    } break;
	case Tempest::WindowEventType::MouseScroll:
	{
		if(!m_Active && wevent.MouseMoved.MouseDeltaX)
			break;

		m_Camera.Offset = std::max(m_Camera.Offset - WheelVelocity*wevent.MouseMoved.MouseDeltaX, 0.0f);
		m_Movement |= MOVEMENT_FLUSH;
	} break;
    case Tempest::WindowEventType::KeyPressed:
    {
        if(!m_Active)
            break;
        switch(wevent.Key)
        {
        case Tempest::KeyboardKey::Up:
        case Tempest::KeyboardKey::Key_W:
        {
            m_Movement |= MOVEMENT_FORWARD;
        } break;
        case Tempest::KeyboardKey::Left:
        case Tempest::KeyboardKey::Key_A:
        {
            m_Movement |= MOVEMENT_LEFT;
        } break;
        case Tempest::KeyboardKey::Right:
        case Tempest::KeyboardKey::Key_D:
        {
            m_Movement |= MOVEMENT_RIGHT;
        } break;
        case Tempest::KeyboardKey::Down:
        case Tempest::KeyboardKey::Key_S:
        {
            m_Movement |= MOVEMENT_BACKWARD;
        } break;
        case Tempest::KeyboardKey::Key_C:
        case Tempest::KeyboardKey::LCtrl:
        {
            m_Movement |= MOVEMENT_ROTATION;
        } break;
        case Tempest::KeyboardKey::Key_R:
        {
            m_Movement |= MOVEMENT_UP;
        } break;
        case Tempest::KeyboardKey::Key_F:
        {
            m_Movement |= MOVEMENT_DOWN;
        } break;
		case Tempest::KeyboardKey::Key_Q:
		{
			m_Movement |= MOVEMENT_ZOOM_IN;
		} break;
        case Tempest::KeyboardKey::Key_E:
		{
			m_Movement |= MOVEMENT_ZOOM_OUT;
		} break;
        default: break;
        }
    } break;
    case Tempest::WindowEventType::KeyReleased:
    {
        if(!m_Active)
            break;
        switch(wevent.Key)
        {
        case Tempest::KeyboardKey::Up:
        case Tempest::KeyboardKey::Key_W:
        {
            m_Movement &= ~MOVEMENT_FORWARD;
        } break;
        case Tempest::KeyboardKey::Left:
        case Tempest::KeyboardKey::Key_A:
        {
            m_Movement &= ~MOVEMENT_LEFT;
        } break;
        case Tempest::KeyboardKey::Right:
        case Tempest::KeyboardKey::Key_D:
        {
            m_Movement &= ~MOVEMENT_RIGHT;
        } break;
        case Tempest::KeyboardKey::Down:
        case Tempest::KeyboardKey::Key_S:
        {
            m_Movement &= ~MOVEMENT_BACKWARD;
        } break;
        case Tempest::KeyboardKey::Key_C:
        case Tempest::KeyboardKey::LCtrl:
        {
            m_Movement &= ~MOVEMENT_ROTATION;
        } break;
        case Tempest::KeyboardKey::Key_R:
        {
            m_Movement &= ~MOVEMENT_UP;
        } break;
        case Tempest::KeyboardKey::Key_F:
        {
            m_Movement &= ~MOVEMENT_DOWN;
        } break;
		case Tempest::KeyboardKey::Key_Q:
		{
			m_Movement &= ~MOVEMENT_ZOOM_IN;
		} break;
        case Tempest::KeyboardKey::Key_E:
		{
			m_Movement &= ~MOVEMENT_ZOOM_OUT;
		} break;
		case Tempest::KeyboardKey::Key_M:
		{
			m_RTSystem.toggleFpsCounter();
		} break;
		case Tempest::KeyboardKey::PrintScreen:
		{
			std::stringstream ss;
			ss << "screenshot.png";
			uint32_t counter = 0;
			for(;;)
			{
				if(!System::Exists(ss.str()))
					break;
				ss.str("");
				ss << "screenshot_" << counter++ << ".png";
			}
			auto backbuffer = m_RTSystem.getLastFrameTexture();
			Tempest::SaveImage(backbuffer->getHeader(), backbuffer->getData(), Path(ss.str()));
			Log(LogLevel::Info, "Saved screenshot: ", ss.str());
		} break;
		default: break;
        }
    } break;
    }
}

template<class TRTSystem>
bool RayTracingView<TRTSystem>::render()
{
    auto& window = m_RTSystem.getWindow();
    
    Tempest::Vector2 window_size{ float(window.getWidth() - 1), float(window.getHeight() - 1) };
    auto cur_time = m_Timer.time();
    
    float elapsed_time = (float)(cur_time - m_PreviousTime);
    if(m_Movement)
    {
		Tempest::Vector3 left, forward;
		Tempest::FastSinCos(-m_Camera.Yaw, &left.z, &left.x);
		left.y = 0.0f;
		forward = { -left.z, 0.0f, left.x };

		auto rel_cursor = m_MousePosition - m_PreviousMousePosition;

		if(m_Movement & MOVEMENT_DRAG)
			m_Camera.Base += DragVelocity*(rel_cursor.x*left + rel_cursor.y*forward);
        if(m_Movement & MOVEMENT_FORWARD)
            m_Camera.Base += forward*Velocity*elapsed_time;
        if(m_Movement & MOVEMENT_BACKWARD)
            m_Camera.Base -= forward*Velocity*elapsed_time;
        if(m_Movement & MOVEMENT_RIGHT)
            m_Camera.Base -= left*Velocity*elapsed_time;
        if(m_Movement & MOVEMENT_LEFT)
            m_Camera.Base += left*Velocity*elapsed_time;
        if(m_Movement & MOVEMENT_UP)
            m_Camera.Base.y -= Velocity*elapsed_time;
        if(m_Movement & MOVEMENT_DOWN)
            m_Camera.Base.y += Velocity*elapsed_time;
		if(m_Movement & MOVEMENT_ZOOM_IN)
			m_Camera.Base.z = std::max(m_Camera.Offset - Velocity*elapsed_time, 0.0f);
		if(m_Movement & MOVEMENT_ZOOM_OUT)
			m_Camera.Base.z += Velocity*elapsed_time;

        if(m_Movement & MOVEMENT_ROTATION)
        {
			m_Camera.Yaw -= MouseSpeed*rel_cursor.x;
			m_Camera.Roll -= MouseSpeed*rel_cursor.y;

			m_Camera.Roll = Clampf(m_Camera.Roll, 0.0f, Tempest::MathPi*0.5f);
        }

        m_ViewProjectionInverse = ComputeViewProjectionInverse(m_Camera);
		m_Movement &= ~MOVEMENT_FLUSH;
    }
    
    m_PreviousMousePosition = m_MousePosition;
    
    m_RTSystem.completeFrameAndRestart(m_Width, m_Height, m_ViewProjectionInverse);

    m_PreviousTime = cur_time;
    return m_RTSystem.presentFrame();
}

template<class TRTSystem>
void RayTracingView<TRTSystem>::view()
{
    Tempest::WindowSystemEvent wevent;

    bool alive = true;

    auto& window = m_RTSystem.getWindow();

    do
    {
        m_RTSystem.completeFrame();

        for(;;)
        {
            auto status = window.getEvent(&wevent);
			if(!status)
                break;

            this->handleEvent(wevent);
        }

        alive = this->render();
    } while(alive);
}

template class RayTracingView<RayTracingSystem>;

#ifndef DISABLE_CUDA
    template class RayTracingView<RayTracingCudaSystem>;
#endif
}