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

#ifndef _TEMPEST_EQUIRECTANGULAR_MAP_HH_
#define _TEMPEST_EQUIRECTANGULAR_MAP_HH_

#include "tempest/graphics/texture.hh"
#include "tempest/graphics/sampling-wrapper.hh"

namespace Tempest
{
// Refer to this website for more information: http://gl.ict.usc.edu/Data/HighResProbes/
inline Vector2 CartesianToEquirectangularCoordinates(const Vector3& dir)
{
    return { (1.0f + atan2f(dir.x, -dir.z)/MathPi)*0.5f, 1.0f - acosf(dir.y)/MathPi };
}

inline Vector3 EquirectangularToCartesianCoordinates(const Vector2& tc)
{
    float theta = MathPi*(tc.x*2.0f - 1.0f);
    float phi = MathPi*tc.y;

    float sin_theta, cos_theta;
    FastSinCos(theta, &sin_theta, &cos_theta);

    float sin_phi, cos_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);

    return { sin_phi*sin_theta, -cos_phi, -sin_phi*cos_theta };
}

class EquirectangularMap
{
    const void* m_Texture;
public:
    EquirectangularMap(const void* tex)
        :   m_Texture(tex)
    {
    }

    Spectrum sampleSpectrum(const Vector3& dir) const
    {
        Vector2 tc = CartesianToEquirectangularCoordinates(dir);
        return Tempest::SampleSpectrum(m_Texture, tc);
    }

    Vector4 sampleRGBA(const Vector3& dir) const
    {
        Vector2 tc = CartesianToEquirectangularCoordinates(dir);
        return Tempest::SampleRGBA(m_Texture, tc);
    }

    Vector3 sampleRGB(const Vector3& dir) const
    {
        Vector2 tc = CartesianToEquirectangularCoordinates(dir);
        return Tempest::SampleRGB(m_Texture, tc);
    }
};

class CubeMap;
CubeMap* ConvertEquirectangularMapToCubeMap(const TextureDescription& cube_tex_desc, const EquirectangularMap& eqrect_map);
}

#endif // _TEMPEST_EQUIRECTANGULAR_MAP_HH_