/*   The MIT License
*
*   Tempest Renderer
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

#ifndef _BVH_INTERSECT_LEAF_HH_
#define _BVH_INTERSECT_LEAF_HH_

namespace Tempest
{
struct RayIntersectData
{
    Vector3 Direction,
            Origin;
    float   Near,
            Far;
};

inline EXPORT_CUDA bool IntersectPrimBVH(const RayIntersectData& intersect, const AABBUnaligned& box)
{
    float tmin, tmax;
    return IntersectRayAABB(intersect.Direction, intersect.Origin, intersect.Near, intersect.Far, box.MinCorner, box.MaxCorner, &tmin, &tmax);
}

inline EXPORT_CUDA bool IntersectPrimBVH(const Vector3& pos, const AABBUnaligned& box)
{
    return IntersectPointAABB(pos, box);
}

inline EXPORT_CUDA bool IntersectPrimBVH(const Vector2& pos, const AABB2& box)
{
    return IntersectPointAABB(pos, box);
}

inline EXPORT_CUDA bool IntersectPrimBVH(const Sphere& pos, const AABBUnaligned& box)
{
    return IntersectSphereAABB(pos, box);
}

inline EXPORT_CUDA bool IntersectPrimBVH(const AABB2& lhs, const AABB2& rhs)
{
    return IntersectAABBAABB(lhs, rhs);
}

struct IntersectRectSetQuery
{
    const Rect3*    Rects;
    float           IntersectDistance;
    uint32_t        PrimitiveID;
    Vector3         Normal;
    Vector2         BarycentricCoordinates;

    inline EXPORT_CUDA bool operator()(uint32_t prim_id, const RayIntersectData& ray)
    {
        auto& rect = Rects[prim_id];

        Vector2 barycentric;
        Vector3 normal;
        float dist;
        auto status = IntersectRect3(ray.Direction, ray.Origin, rect, &dist, &barycentric.x, &barycentric.y, &normal);
        if(status && dist < IntersectDistance && Dot(normal, ray.Direction) < 0.0f)
        {
            Normal = normal;
            BarycentricCoordinates = barycentric;
            IntersectDistance = dist;
            PrimitiveID = prim_id;
            return true;
        }
        else
        {
            return false;
        }
    }
};


struct IntersectTriangleQuery3DTwoSided
{
    const uint8_t*  Vertices;
    uint32_t        Stride;
    const uint32_t* Indices;
    float           IntersectDistance;
    uint32_t        PrimitiveID;
    Vector3         Normal;
    Vector2         BarycentricCoordinates;

    inline EXPORT_CUDA bool operator()(uint32_t prim_id, const RayIntersectData& ray_intersect)
    {
        auto idx = prim_id*3;
        auto i0 = Indices[idx++];
        auto i1 = Indices[idx++];
        auto i2 = Indices[idx++];

        auto& v0 = *reinterpret_cast<const Vector3*>(Vertices + i0*Stride);
        auto& v1 = *reinterpret_cast<const Vector3*>(Vertices + i1*Stride);
        auto& v2 = *reinterpret_cast<const Vector3*>(Vertices + i2*Stride);

        Vector3 normal;
        Vector2 barycentric;
        float   dist;
        auto status = IntersectRayTriangle(ray_intersect.Direction, ray_intersect.Origin, v0, v1, v2, &normal, &barycentric, &dist);
        float epsilon = 1e-3f;
        if(status && epsilon < IntersectDistance && dist < IntersectDistance)
        {
            Normal = Dot(normal, ray_intersect.Direction) < 0.0f ? normal : -normal;
            BarycentricCoordinates = barycentric;
            IntersectDistance = dist;
            PrimitiveID = prim_id;
        }
        return status;
    }
};

struct IntersectTriangleQuery3DCull
{
    const uint8_t*  Vertices;
    uint32_t        Stride;
    const uint32_t* Indices;
    float           IntersectDistance;
    uint32_t        PrimitiveID;
    Vector3         Normal;
    Vector2         BarycentricCoordinates;

    inline EXPORT_CUDA bool operator()(uint32_t prim_id, const RayIntersectData& ray_intersect)
    {
        auto idx = prim_id*3;
        auto i0 = Indices[idx++];
        auto i1 = Indices[idx++];
        auto i2 = Indices[idx++];

        auto& v0 = *reinterpret_cast<const Vector3*>(Vertices + i0*Stride);
        auto& v1 = *reinterpret_cast<const Vector3*>(Vertices + i1*Stride);
        auto& v2 = *reinterpret_cast<const Vector3*>(Vertices + i2*Stride);

        Vector3 normal;
        Vector2 barycentric;
        float   dist;
        auto status = IntersectRayTriangle(ray_intersect.Direction, ray_intersect.Origin, v0, v1, v2, &normal, &barycentric, &dist);
        if(status && dist < IntersectDistance && Dot(normal, ray_intersect.Direction) < 0.0f)
        {
            Normal = normal;
            BarycentricCoordinates = barycentric;
            IntersectDistance = dist;
            PrimitiveID = prim_id;
            return true;
        }
        else
        {
            return false;
        }
    }
};

struct IntersectTriangleQuery2D
{
    const Vector2*  Vertices;
    const uint32_t* Indices;
    uint32_t        PrimitiveID;
    Vector3         BarycentricCoordinates;

    inline EXPORT_CUDA bool operator()(uint32_t prim_id, const Vector2& pos)
    {
        auto idx = prim_id*3;
        auto i0 = Indices[idx++];
        auto i1 = Indices[idx++];
        auto i2 = Indices[idx++];

        auto& v0 = Vertices[i0];
        auto& v1 = Vertices[i1];
        auto& v2 = Vertices[i2];

        Vector3 barycentric;
        auto status = IntersectTriangle(pos, v0, v1, v2, &barycentric);
        if(status)
        {
            PrimitiveID = prim_id;
            BarycentricCoordinates = barycentric;
        }
        return status;
    }
};
}

#endif // _BVH_INTERSECT_LEAF_HH_
