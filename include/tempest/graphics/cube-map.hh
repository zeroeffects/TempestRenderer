/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 Zdravko Velinov
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

#ifndef __TEMPEST_CUBEMAP_HH__
#define __TEMPEST_CUBEMAP_HH__

#include "tempest/graphics/texture.hh"
#include "tempest/math/vector4.hh"

#include <memory>

namespace Tempest
{
inline Vector2 CartesianToCubeMapCoordinates(const Vector3& dir, size_t* face_id)
{
    float ax = fabsf(dir.x);
    float ay = fabsf(dir.y);
    float az = fabsf(dir.z);
    float s, t, m;
    if(ax > ay)
    {
        if(ax > az)
        {
            s = -dir.z * Sign(dir.x);
            t = -dir.y;
            m = ax;
            *face_id = std::signbit(dir.x);
        }
        else
        {
            s = dir.x * Sign(dir.z);
            t = -dir.y;
            m = az;
            *face_id = 4 + std::signbit(dir.z);
        }
    }
    else
    {
        if(ay > az)
        {
            s = dir.x;
            t = dir.z * Sign(dir.y);
            m = ay;
            *face_id = 2 + std::signbit(dir.y);
        }
        else
        {
            s = dir.x * Sign(dir.z);
            t = -dir.y;
            m = az;
            *face_id = 4 + std::signbit(dir.z);
        }
    }

    return { s/m*0.5f + 0.5f, t/m*0.5f + 0.5f };
}

inline Vector3 CubeMapToCartesianCoordinates(const Vector2& tc, size_t face_id)
{
    auto x = tc.x*2.0f - 1.0f,
         y = tc.y*2.0f - 1.0f,
         z = 1.0f/sqrtf(1.0f + x*x + y*y);

    x *= z;
    y *= z;

    Tempest::Vector3 out_dir;
    if(face_id >= 4)
    {
        float sign = static_cast<float>(1 - 2*(static_cast<int>(face_id) - 4));

        out_dir.x = sign*x;
        out_dir.y = -y;
        out_dir.z = sign*z;
    }
    else if(face_id >= 2)
    {
        float sign = static_cast<float>(1 - 2*(static_cast<int>(face_id) - 2));

        out_dir.x =      x;
        out_dir.y = sign*z;
        out_dir.z = sign*y;
    }
    else
    {
        float sign = static_cast<float>(1 - 2*static_cast<int>(face_id));

        out_dir.x =  sign*z;
        out_dir.y = -y;
        out_dir.z = -sign*x;
    }

    return out_dir;
}

class CubeMap
{
    TextureDescription          m_Description;
    std::unique_ptr<uint8_t[]>  m_Data;
public:
    CubeMap(Texture** textures);
    CubeMap(const TextureDescription& tex_desc, uint8_t* data)
        :   m_Description(tex_desc),
            m_Data(data) {}

    const TextureDescription& getHeader() const { return m_Description; }

    void* getData() { return m_Data.get(); }
    const void* getData() const { return m_Data.get(); }

    Spectrum sampleSpectrum(const Vector3& dir);
    Vector3 sampleRGB(const Vector3& dir);
    Vector4 sampleRGBA(const Vector3& dir);

    Vector3 fetchFaceRGB(uint32_t face, uint32_t x, uint32_t y);
};
}

#endif // __TEMPEST_CUBEMAP_HH__