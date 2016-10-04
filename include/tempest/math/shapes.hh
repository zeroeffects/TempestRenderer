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

#ifndef _TEMPEST_SHAPE_HH_
#define _TEMPEST_SHAPE_HH_

#include "tempest/math/vector3.hh"
#include "tempest/math/matrix3.hh"
#include "tempest/math/quaternion.hh"
#include <cstdint>

namespace Tempest
{
struct AABB2
{
	typedef Vector2 BoundsType;

    BoundsType MinCorner;
    BoundsType MaxCorner;
};

struct AABBUnaligned
{
	typedef Vector3 BoundsType;

    BoundsType MinCorner;
    BoundsType MaxCorner;
};

struct Rect2
{
    Vector2                       Center;
    float                         Orientation;
    Vector2                       Size;
};

struct Rect3
{
    Vector3                       Center;
    Quaternion                    Orientation;
    Vector2                       Size;
};

struct Disk3
{
    Vector3                       Center;
    Vector3                       Normal;
    float                         InnerRadius;
    float                         OuterRadius;
};

struct Disk2
{
    Vector2                       Center;
    float                         InnerRadius;
    float                         OuterRadius;
};

struct Capsule2
{
    Vector2                       Center[2];
    float                         Radius;
};

struct Sphere
{
    Vector3                       Center;
    float                         Radius;
};

struct Ellipsoid
{
    Vector3                       Center;
    Vector3                       Scale;
    Quaternion                    Orientation;
};

struct Cylinder
{
    Vector3                       Center;
    float                         Radius;
    float                         HalfHeight;
};

struct ObliqueCylinder
{
    Cylinder                      CylinderShape;
    Vector2                       TiltDirection;
    float                         Tilt;
};

struct HelixCylinder
{
    Cylinder                      CylinderShape;
    float                         AngularSpeed;
    float                         CurvatureRadius;
};

struct ShapeVertex
{
    Vector3                       Position;
    Vector3                       Normal;
};

// Or you can call it doughnut
struct Torus
{
    Vector3                        Center;
    float                          TubeRadius;
    float                          TorusRadius;
};

typedef Torus Doughnut;
typedef HelixCylinder SpringShape;

void TriangleTessellationNoNormals(const Sphere& sphere, uint32_t sphere_long_tes, uint32_t sphere_lat_tes, Vector3** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count);

void TriangleTessellation(const Sphere& sphere, uint32_t sphere_long_tes, uint32_t sphere_lat_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count);
void TriangleTessellation(const Cylinder& cylinder, uint32_t circular_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count);
void TriangleTessellation(const Cylinder& cylinder, uint32_t circular_tes, uint32_t vert_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count);
void TriangleTessellation(const ObliqueCylinder& cylinder, uint32_t circular_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count);
void TriangleTessellation(const HelixCylinder& cylinder, uint32_t circular_tes, uint32_t vert_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count);

template<class TFunc>
void TriangleTessellation(const Sphere& sphere, uint32_t sphere_long_tes, uint32_t sphere_lat_tes, ShapeVertex** out_verts, uint32_t* vert_count, int32_t** out_indices, uint32_t* indices_count,
                          const TFunc& func)
{
    auto regular_grid_size = (sphere_lat_tes - 1)*sphere_long_tes*6;
    *indices_count = regular_grid_size + sphere_long_tes*6;
    auto indices = *out_indices = new int32_t[*indices_count];

    *vert_count = sphere_lat_tes*sphere_long_tes + 2;
    auto verts = *out_verts = new ShapeVertex[*vert_count];
    
    uint32_t vert_idx = 0,
             idx = 0;

    for(uint32_t latitude = 0; latitude < sphere_lat_tes; ++latitude)
    {
        float cos_theta, sin_theta;

        FastSinCos(MathPi*(latitude + 1)/(sphere_lat_tes + 1), &sin_theta, &cos_theta);
        for(uint32_t longitude = 0; longitude < sphere_long_tes; ++longitude)
        {
            float cos_phi, sin_phi;

            FastSinCos(2.0f*MathPi*longitude/sphere_long_tes, &sin_phi, &cos_phi);

            auto& vert = verts[vert_idx++];

            Tempest::Vector3 norm = { cos_phi*sin_theta, sin_phi*sin_theta, cos_theta };

            float scale = func(norm);
            TGE_ASSERT(scale <= 1.0f, "Perturbations should be smaller than the bounding sphere");

            vert.Position = sphere.Center + norm*sphere.Radius*scale;
            vert.Normal = {};

            if(latitude != sphere_lat_tes - 1)
            {
                int32_t next_vertex = longitude != sphere_long_tes - 1 ? 1 : -(int32_t)(sphere_long_tes - 1);

                uint32_t base_idx = latitude*sphere_long_tes + longitude;
                indices[idx++] = base_idx;
                indices[idx++] = base_idx + next_vertex + sphere_long_tes;
                indices[idx++] = base_idx + next_vertex;
                indices[idx++] = base_idx;
                indices[idx++] = base_idx + sphere_long_tes;
                indices[idx++] = base_idx + next_vertex + sphere_long_tes;
            }
        }
    }

    {
    auto& vert = verts[vert_idx++];
    Tempest::Vector3 norm = { 0.0f, 0.0f,  1.0f };
    float scale = func(norm);
    TGE_ASSERT(scale <= 1.0f, "Perturbations should be smaller than the bounding sphere");
    vert.Position = sphere.Center + norm*sphere.Radius*scale;
    vert.Normal = {};
    }
    
    {
    auto& vert = verts[vert_idx++];
    Tempest::Vector3 norm = { 0.0f, 0.0f, -1.0f };
    float scale = func(norm);
    TGE_ASSERT(scale <= 1.0f, "Perturbations should be smaller than the bounding sphere");
    vert.Position = sphere.Center + norm*sphere.Radius*scale;
    vert.Normal = {};
    }

    TGE_ASSERT(idx == regular_grid_size, "Invalid index count of the main grid");
    TGE_ASSERT(vert_idx == *vert_count, "Invalid vertex population");

    for(uint32_t longitude = 0; longitude < sphere_long_tes; ++longitude)
    {
        int32_t next_vertex = longitude != sphere_long_tes - 1 ? 1 : -(int32_t)(sphere_long_tes - 1);

        indices[idx++] = sphere_lat_tes*sphere_long_tes;
        indices[idx++] = longitude;
        indices[idx++] = longitude + next_vertex;

        indices[idx++] = sphere_lat_tes*sphere_long_tes + 1;
        uint32_t base_idx = (sphere_lat_tes - 1)*sphere_long_tes;
        indices[idx++] = base_idx + longitude;
        indices[idx++] = base_idx + longitude + next_vertex;
    }

    TGE_ASSERT(idx == *indices_count, "Invalid index count");

    auto count = *indices_count;
    for(size_t idx = 0; idx < count;)
    {
        auto prev_idx = indices[idx++];
        auto current_idx = indices[idx++];
        auto next_idx = indices[idx++];
        
        auto& prev = verts[prev_idx];
        auto& current = verts[current_idx];
        auto& next = verts[next_idx];
        auto d0 = prev.Position - current.Position;
        auto d1 = next.Position - current.Position;
        Vector3 norm = Cross(d1, d0);
        prev.Normal += norm;
        current.Normal += norm;
        next.Normal += norm;
    }
    
    count = *vert_count;
    for(size_t idx = 0; idx < count; ++idx)
    {
        NormalizeSelf(&verts[idx].Normal);
    }
}

void CurveSkewYZ(const ShapeVertex* in_verts, uint32_t vert_count, float magnitude, float max_len, ShapeVertex* out_verts);

// NOTE: if we return AABBUnaligned it doesn't work in both CUDA and MSVC
inline EXPORT_CUDA void Rect3Bounds(const Rect3& rect, AABBUnaligned* result)
{
    auto extend = Vector3Abs(rect.Size.x*ToTangent(rect.Orientation)) + Vector3Abs(rect.Size.y*ToBinormal(rect.Orientation));

    result->MinCorner = rect.Center - extend;
    result->MaxCorner = rect.Center + extend;
}

inline EXPORT_CUDA void Disk3Bounds(const Disk3& disk, AABBUnaligned* result)
{
    // Make an axis aligned basis
    Vector3 d0 = Cross(disk.Normal, Vector3{1.0f, 0.0f, 0.0f}),
			d1 = Cross(disk.Normal, Vector3{0.0f, 1.0f, 0.0f});
    float d0_len = Dot(d0, d0);
    float d1_len = Dot(d1, d1);

    Vector3 tan = d0_len > d1_len ? d0 / sqrtf(d0_len) : d1 / sqrtf(d1_len);
	Vector3 binorm = Cross(disk.Normal, tan);

    Vector3 tan_extend = Vector3Abs(tan*disk.OuterRadius);
    Vector3 binorm_extend = Vector3Abs(binorm*disk.OuterRadius);

    Vector3 max_extend = Vector3Max(tan_extend, binorm_extend);

    result->MinCorner = disk.Center - max_extend;
    result->MaxCorner = disk.Center + max_extend;
}

inline EXPORT_CUDA void SphereBounds(const Sphere& sphere, AABBUnaligned* result)
{
	result->MinCorner = sphere.Center - sphere.Radius;
	result->MaxCorner = sphere.Center + sphere.Radius;
}

inline EXPORT_CUDA void Rect2Bounds(const Matrix2& rotate_scale, const Vector2& pos, AABB2* result)
{
    Vector2 tangent_extend = Vector2Abs(rotate_scale.column(0));
    Vector2 binorm_extend = Vector2Abs(rotate_scale.column(1)); // norm in 2d space

    auto extends = (tangent_extend + binorm_extend)*1.001f;

    result->MinCorner = pos - extends; // TODO: slightly extend to hide floating point precision artifacts
    result->MaxCorner = pos + extends;
}

inline EXPORT_CUDA void TriangleBounds(const Vector2& v0, const Vector2& v1, const Vector2& v2, AABB2* result)
{
    auto min = v0;
    auto max = v1;
    if(v0.x > v1.x)
        Swap(min.x, max.x);
    if(v0.y > v1.y)
        Swap(min.y, max.y);
    if(v2.x < min.x)
        min.x = v2.x;
    else if(v2.x > max.x)
        max.x = v2.x;
    if(v2.y < min.y)
        min.y = v2.y;
    else if(v2.y > max.y)
        max.y = v2.y;
    result->MinCorner = min;
    result->MaxCorner = max;
}

inline EXPORT_CUDA void TriangleBounds(const Vector3& v0, const Vector3& v1, const Vector3& v2, AABBUnaligned* result)
{
    auto min = v0;
    auto max = v1;
    if(v0.x > v1.x)
        Swap(min.x, max.x);
    if(v0.y > v1.y)
        Swap(min.y, max.y);
	if(v0.z > v1.z)
		Swap(min.z, max.z);

    if(v2.x < min.x)
        min.x = v2.x;
    else if(v2.x > max.x)
        max.x = v2.x;
    if(v2.y < min.y)
        min.y = v2.y;
    else if(v2.y > max.y)
        max.y = v2.y;
	if(v2.z < min.z)
		min.z = v2.z;
	else if(v2.z > max.z)
		max.z = v2.z;
    result->MinCorner = min;
    result->MaxCorner = max;
}

inline EXPORT_CUDA float Area(const AABB2& aabb)
{
    Vector2 extend = aabb.MaxCorner - aabb.MinCorner;
    return extend.x*extend.y;
}

inline EXPORT_CUDA bool operator==(const AABB2& lhs, const AABB2& rhs)
{
	return lhs.MinCorner == rhs.MinCorner &&
		   lhs.MaxCorner == rhs.MaxCorner;
}
}

#endif // _TEMPEST_SHAPE_HH_
