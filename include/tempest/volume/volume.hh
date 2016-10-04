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

#ifndef _TEMPEST_VOLUME_HH_
#define _TEMPEST_VOLUME_HH_

#include <cstdint>
#include "tempest/math/functions.hh"
#include "tempest/math/vector3.hh"

namespace Tempest
{
struct Box
{
    int32_t X, Y, Z;
};

inline Vector3 ToVector3(const Box& box)
{
    return Vector3{static_cast<float>(box.X), static_cast<float>(box.Y), static_cast<float>(box.Z)};
}

inline bool operator==(const Box& lhs, const Box& rhs)
{
    return lhs.X == rhs.X &&
           lhs.Y == rhs.Y &&
           lhs.Z == rhs.Z;
}

inline Vector3 operator*(const Box& box, const Vector3& scale)
{
    return Vector3{box.X*scale.x, box.Y*scale.y, box.Z*scale.z};
}

enum class GridDataType: uint32_t
{
    Invalid = 0,
    Float32 = 1,
	Float16 = 2,
	UInt8 = 3,
	QuantizedDirections = 4
};

struct Volume
{
    Box     Dimensions;
    void*   Data = nullptr;

    ~Volume();
};

struct VolumeRoot
{
    Vector3         MinCorner;
    Vector3         MaxCorner;
    Box             Dimensions = Box{ 0, 0, 0 };
    
    Volume*         Volumes;

    ~VolumeRoot();
};

VolumeRoot* ParseVolumeHierarchy(const std::string& name);
VolumeRoot* ParseVolume(const std::string& density, const std::string& orientation);
}

#endif // _TEMPEST_VOLUME_HH_