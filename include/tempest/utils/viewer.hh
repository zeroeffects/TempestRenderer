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

#ifndef _TEMPEST_VIEWER_HH_
#define _TEMPEST_VIEWER_HH_

#include "tempest/math/matrix4.hh"
#include "tempest/math/matrix3.hh"
#include "tempest/utils/system.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/timer.hh"
#include "tempest/graphics/os-window.hh"
#include "tempest/image/image.hh"

const float Velocity = 10e-7f;
const float WheelVelocity = 1e-3f;
const float DragVelocity = 1e-2f;
const float MouseSpeed = 1e-2f;

namespace Tempest
{
struct FreeCamera
{
    float   Yaw = 0.0f,
            Roll = 0.0f,
            Offset = 10.0f;
    Vector3 Base = {};
    Matrix4 Projection;
};

inline Matrix4 ComputeViewProjectionInverse(FreeCamera& cam)
{
    Tempest::Matrix4 view;
    view.identity();
    view.translate(Tempest::Vector3{ 0.0f, 0.0f, -cam.Offset });
	view.rotateX(Tempest::MathPi*0.5f-cam.Roll);
    view.rotateY(-cam.Yaw);
    view.translate(cam.Base);

    Tempest::Matrix4 view_proj = cam.Projection*view;
    return view_proj.inverse();
}

template<class TRTSystem>
class RayTracingView
{
    enum
    {
        MOVEMENT_FORWARD  = 1 << 0,
        MOVEMENT_LEFT     = 1 << 1,
        MOVEMENT_RIGHT    = 1 << 2,
        MOVEMENT_BACKWARD = 1 << 3,
        MOVEMENT_ROTATION = 1 << 4,
        MOVEMENT_UP       = 1 << 5,
        MOVEMENT_DOWN     = 1 << 6,
		MOVEMENT_ZOOM_IN  = 1 << 7,
		MOVEMENT_ZOOM_OUT = 1 << 8,
		MOVEMENT_FLUSH    = 1 << 9,
		MOVEMENT_DRAG     = 1 << 10,
    };

    TRTSystem&              m_RTSystem;
    uint32_t                m_Movement = 0;
    bool                    m_Active = true; // TODO: get window status
    Vector2                 m_MousePosition,
                            m_PreviousMousePosition;
    uint32_t                m_Width,
                            m_Height;
    TimeQuery               m_Timer;
    uint64_t                m_PreviousTime;
    Matrix4                 m_ViewProjectionInverse;

    FreeCamera&             m_Camera;

public:
    RayTracingView(TRTSystem& rt_sys, uint32_t image_width, uint32_t image_height, FreeCamera& cam);
    void handleEvent(Tempest::WindowSystemEvent wevent);
    bool render();
    void view();
};

template<class TRTSystem>
void RayTracingViewer(TRTSystem& rt_sys, uint32_t image_width, uint32_t image_height, FreeCamera& cam)
{
    RayTracingView<TRTSystem> viewer(rt_sys, image_width, image_height, cam);

    viewer.view();
}
}

#endif // _TEMPEST_VIEWER_HH_