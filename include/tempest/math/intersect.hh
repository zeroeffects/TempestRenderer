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

#ifndef _TEMPEST_INTERSECT_HH_
#define _TEMPEST_INTERSECT_HH_

#include "tempest/math/shapes.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/math/quaternion.hh"

namespace Tempest
{
#if defined(__CUDA_ARCH__) || (defined(LINUX)  && __CUDACC__)

#ifdef LINUX
#   define EXPORT_PARAMS __device__ __host__
#else
#   define EXPORT_PARAMS __device__
#endif

inline bool EXPORT_PARAMS IntersectRayAABB(const Vector3& in_dir, const Vector3& in_org, float in_tnear, float in_tfar, const Vector3& in_min_corner, const Vector3& in_max_corner, float *out_tmin, float *out_tmax)
{
    Vector3 rcp_dir = 1.0f/in_dir;

    Vector3 tmin_v3 = SelectGE(rcp_dir, 0.0f, in_min_corner, in_max_corner),
            tmax_v3 = SelectGE(rcp_dir, 0.0f, in_max_corner, in_min_corner);

    Vector3 epsilon = ToVector3(FLT_MIN);

    tmin_v3 = (tmin_v3 - in_org + epsilon)*rcp_dir;
    tmax_v3 = (tmax_v3 - in_org + epsilon)*rcp_dir;

    float global_min = MaxValue(tmin_v3);
    float global_max = MinValue(tmax_v3);

    if(global_min > global_max || global_max < in_tnear || global_min > in_tfar)
        return false;

    *out_tmin = global_min;
    *out_tmax = global_max;
    return true;
}
#else
inline bool IntersectRayAABB(const Vector3& in_dir, const Vector3& in_org, float in_tnear, float in_tfar, const Vector3& in_min_corner, const Vector3& in_max_corner, float *out_tmin, float *out_tmax)
{
    __m128 dir_v4 = _mm_setr_ps(in_dir.x, in_dir.y, in_dir.z, 0.0f);
    __m128 org = _mm_setr_ps(in_org.x, in_org.y, in_org.z, 0.0f);
    //__m128 rcp_dir = _mm_rcp_ps(dir_v4); // poor precision
    __m128 rcp_dir = _mm_div_ps(_mm_set1_ps(1), dir_v4);

    __m128 mask = _mm_cmpge_ps(rcp_dir, _mm_setzero_ps()); // Might not be perfect
    __m128 min_corner = _mm_setr_ps(in_min_corner.x, in_min_corner.y, in_min_corner.z, 0.0f);
    __m128 max_corner = _mm_setr_ps(in_max_corner.x, in_max_corner.y, in_max_corner.z, 0.0f);
    __m128 tmin_v4 = _mm_blendv_ps(max_corner, min_corner, mask);
    __m128 tmax_v4 = _mm_blendv_ps(min_corner, max_corner, mask);

    __m128 min_value = _mm_set1_ps(std::numeric_limits<float>::min());
    tmin_v4 = _mm_sub_ps(tmin_v4, org);
    tmin_v4 = _mm_add_ps(tmin_v4, min_value);
    tmin_v4 = _mm_mul_ps(tmin_v4, rcp_dir);
    //tmin_v4 = _mm_div_ps(tmin_v4, dir_v4);
    tmin_v4 = _mm_max_ps(tmin_v4, _mm_permute_ps(tmin_v4, _MM_SHUFFLE(3, 0, 0, 1)));
    tmin_v4 = _mm_max_ps(tmin_v4, _mm_permute_ps(tmin_v4, _MM_SHUFFLE(3, 1, 2, 2)));

    tmax_v4 = _mm_sub_ps(tmax_v4, org);
    tmax_v4 = _mm_add_ps(tmax_v4, min_value);
    tmax_v4 = _mm_mul_ps(tmax_v4, rcp_dir);
    //tmax_v4 = _mm_div_ps(tmax_v4, dir_v4);
    tmax_v4 = _mm_min_ps(tmax_v4, _mm_permute_ps(tmax_v4, _MM_SHUFFLE(3, 0, 0, 1)));
    tmax_v4 = _mm_min_ps(tmax_v4, _mm_permute_ps(tmax_v4, _MM_SHUFFLE(3, 1, 2, 2)));

    float tmin = ElementFloat<0>(tmin_v4);
    float tmax = ElementFloat<0>(tmax_v4);
    if(tmin > tmax || tmax < in_tnear || tmin > in_tfar)
        return false;

    *out_tmin = tmin;
    *out_tmax = tmax;
    return true;
}
#endif

inline EXPORT_CUDA bool IntersectRect3(const Vector3& dir, const Vector3& org, const Rect3& rect, float *out_t, float *out_u, float *out_v, Vector3* out_norm)
{
    Matrix3 orient = ToMatrix3(rect.Orientation);
    const Vector3& norm = orient.column(2);
    float denom = Dot(dir, norm);
    if(denom > -TEMPEST_WEAK_FLOAT_EPSILON)
        return false;
    
    Vector3 dist_to_plane = rect.Center - org;

    float intersect = Dot(dist_to_plane, norm)/denom;
    if(intersect < 0.0f)
        return false;

    auto plane_pos = (intersect*dir + org) - rect.Center;
    float x = Dot(plane_pos, orient.column(0));
    float y = Dot(plane_pos, orient.column(1));
    
    if(fabsf(x) >= rect.Size.x ||
       fabsf(y) >= rect.Size.y)
        return false;

    *out_t = intersect;
    *out_u = (x/rect.Size.x)*0.5f + 0.5f;
    *out_v = (y/rect.Size.y)*0.5f + 0.5f;
	*out_norm = orient.column(2);

    return true;
}

inline EXPORT_CUDA bool IntersectDisk3(const Vector3& dir, const Vector3& org, const Disk3& disk, float *out_t, float *out_u, float *out_v, Vector3* out_norm)
{
    const Vector3& norm = disk.Normal;
    float denom = Dot(dir, norm);
    if(denom > -TEMPEST_WEAK_FLOAT_EPSILON)
        return false;
    
    Vector3 dist_to_plane = disk.Center - org;

    float intersect = Dot(dist_to_plane, norm)/denom;
    if(intersect < 0.0f)
        return false;

    auto plane_pos = disk.Center - (intersect*dir + org);
    
    float dist_from_center = Length(plane_pos);

    if(disk.InnerRadius > dist_from_center || dist_from_center > disk.OuterRadius)
        return false;

    *out_t = intersect;
    *out_u = 0.0f; // TODO
    *out_v = 0.0f;
	*out_norm = disk.Normal;

    return true;
}

inline EXPORT_CUDA bool IntersectSphere(const Vector3& dir, const Vector3& org, const Sphere& sphere, float* out_t, float* out_u, float* out_v, Vector3* out_norm)
{
    Vector3 dist_to_center = sphere.Center - org;
    float b = Dot(dist_to_center, dir);
    float radius = sphere.Radius;
	float c = Dot(dist_to_center, dist_to_center) - radius*radius;

    float D = b*b - c /* * dir.dot(dir) */;

    if(D < 0.0f || b < 0.0f)
    {
        return false;
    }

	float sqrt_D = sqrtf(D);
    float t = b + (2.0f*(b < sqrt_D) - 1.0f)*sqrt_D;

    Vector3 norm = t*dir - dist_to_center;
    NormalizeSelf(&norm);

    if(Dot(norm, dir) > 0.0f)
        return false;

	*out_t = t;
	*out_u = *out_v = 0.0f;
	*out_norm = norm;
	return true;
}

inline EXPORT_CUDA bool IntersectEllipsoid(const Vector3& dir, const Vector3& org, const Ellipsoid& ellipsoid, float* out_t, float* out_u, float* out_v, Vector3* out_norm)
{
    auto inv_transform = Inverse(ellipsoid.Orientation);
    auto dir_elspace = Transform(inv_transform, dir)/ellipsoid.Scale;
    auto dist_to_center_elspace = Transform(inv_transform, ellipsoid.Center - org)/ellipsoid.Scale;

	float a = Dot(dir_elspace, dir_elspace);
    float b = Dot(dist_to_center_elspace, dir_elspace);
	float c = Dot(dist_to_center_elspace, dist_to_center_elspace) - 1.0f;

    float D = b*b - c*a;

    if(D < 0.0f || b < 0.0f)
    {
        return false;
    }

	float sqrt_D = sqrtf(D);
    float t = (b + (2.0f*(b < sqrt_D) - 1.0f)*sqrt_D)/a;

    Vector3 norm = Transform(ellipsoid.Orientation, (t*dir_elspace - dist_to_center_elspace)*ellipsoid.Scale);
    NormalizeSelf(&norm);

    if(Dot(norm, dir) > 0.0f)
        return false;

	*out_t = t;
	*out_u = *out_v = 0.0f;
	*out_norm = norm;
	return true;
}

// That's another spin on the slab test
inline EXPORT_CUDA bool IntersectLineRect2(const Vector2& dir, const Vector2& org, const Matrix2& inv_rotate_scale, const Vector2& pos, float* out_tmin, float* out_tmax)
{
    Vector2 dir_rect_space = inv_rotate_scale.transform(dir); // inv inv transpose
    Vector2 org_rect_space = inv_rotate_scale.transform(org - pos);
    Vector2 inv_dir = 1.0f/dir_rect_space;

    Vector2 max_coord{ Sign(inv_dir.x), Sign(inv_dir.y) };
    Vector2 tmin_v2 = (-max_coord - org_rect_space)*inv_dir;
    Vector2 tmax_v2 = (max_coord - org_rect_space)*inv_dir;

    float tmin = MaxValue(tmin_v2);
    float tmax = MinValue(tmax_v2);

    if(tmax < tmin)
        return false;

    *out_tmin = tmin;
    *out_tmax = tmax;
 
    return true;
}

inline EXPORT_CUDA bool IntersectLineCircleInPlane(const Vector2& dir, const Vector2& org, const Vector2& circleOrg, const float& circleRadius, float* out_tmin, float* out_tmax)
{
    float discriminant, tmp_q, tmp_p_half;
    //Shift coordinate system to have the circleOrg in the origin
    Vector2 tmp_org = org - circleOrg;

    tmp_p_half = Dot(tmp_org, dir);
    tmp_q = Length(tmp_org) * Length(tmp_org) - circleRadius * circleRadius;
    discriminant = tmp_p_half * tmp_p_half - tmp_q;

    //Two intersection points
    if (discriminant > 0.0f) {

        float tmin, tmax, tmp_t;

        tmp_t = -tmp_p_half + sqrtf(discriminant);
        tmin = tmp_t;

        tmp_t = -tmp_p_half - sqrtf(discriminant);

        if (fabsf(tmp_t) >= fabsf(tmin)) {
            tmax = tmp_t;
        }
        else {
            tmax = tmin;
            tmin = tmp_t;
        }
        *out_tmin = tmin;
        *out_tmax = tmax;

        return true;
    }
    //Tangent intersection
    else if (discriminant == 0.0f) {

        float tmin = -tmp_p_half;
        *out_tmin = tmin;
        *out_tmax = tmin;

        return true;

    }
    //No intersection
    else {
        return false;
    }
}

inline EXPORT_CUDA bool IntersectRectCircleInPlane(const Vector2& rectOrg, const Vector2& rectSize, const Vector2& rectDir,
                                                   const Vector2& circleOrg, const float& radius, float* intersectionArea)
{
    //rectSize = {rectWidt, rectLength}
    float tmp_rotationAngle = 0.0f;
    float tmp_intersectionArea;

    Vector2 tmp_rectOrg = rectOrg - circleOrg;

    tmp_rotationAngle = (rectDir.x != 0.0f) ? atan2f(rectDir.y, rectDir.x) : MathPi * 0.5f;
    tmp_rotationAngle = (tmp_rotationAngle >= 0.0f) ? tmp_rotationAngle : (MathPi + tmp_rotationAngle);

    if (tmp_rotationAngle < 0.5f * MathPi) {
        tmp_rotationAngle = 0.5f * MathPi - tmp_rotationAngle;
    }
    else {
        tmp_rotationAngle = - (0.5f * MathPi - (MathPi - tmp_rotationAngle));
    }
    if (rectDir.y == 0.0f) {
        //If the rect is already axis aligned, use Length as x coord, Width as y coord
        tmp_intersectionArea = circle_intersectionArea(tmp_rectOrg.x - 0.5f * rectSize.y, tmp_rectOrg.x + 0.5f * rectSize.y,
                                                       tmp_rectOrg.y - 0.5f * rectSize.x, tmp_rectOrg.y + 0.5f * rectSize.x,
                                                       radius);
    }
    else if (rectDir.x == 0.0f) {
        //If the rect is already axis aligned, use Width as x coord, length as y coord
        tmp_intersectionArea = circle_intersectionArea(tmp_rectOrg.x - 0.5f * rectSize.x, tmp_rectOrg.x + 0.5f * rectSize.x,
                                                       tmp_rectOrg.y - 0.5f * rectSize.y, tmp_rectOrg.y + 0.5f * rectSize.y,
                                                       radius);
    }
    else {
        //Else rotate the origin around the z-axis to achieve axis alignment
        Matrix2 rotMat;
        rotMat.identity();
        rotMat.rotate(tmp_rotationAngle);
        tmp_rectOrg = rotMat.transform(tmp_rectOrg);

        TGE_ASSERT(fabsf(rotMat.transform(rectDir).x) <= 1e-4f, "Broken rotation");

        tmp_intersectionArea = circle_intersectionArea(tmp_rectOrg.x - 0.5f * rectSize.x, tmp_rectOrg.x + 0.5f * rectSize.x,
                                                       tmp_rectOrg.y - 0.5f * rectSize.y, tmp_rectOrg.y + 0.5f * rectSize.y,
                                                       radius);
    }
    *intersectionArea = tmp_intersectionArea;
    TGE_ASSERT(tmp_intersectionArea >= -1e-3f, "intersection Area is not allowed to be negative");
    if (tmp_intersectionArea <= 0.0f) {
        return false;
    }

    return true;
}

inline EXPORT_CUDA bool IntersectRayTriangleMollerTrumbore(const Vector3& dir, const Vector3& org, const Vector3& v0, const Vector3& v1, const Vector3& v2, Vector3* out_normal, Vector2* partial_barycentric_coordinates, float* out_t)
{
    Vector3 e0 = v1 - v0,
            e1 = v2 - v0;

    auto p = Cross(dir, e1);
    float det = Dot(e0, p);
    // sliding
    const float epsilon = 1e-5f;
    if(-epsilon < det && det < epsilon) // fabsf?
        return false;

    float inv_det = 1.0f/det;
    auto dist_to_v0 = org - v0;

    float u = Dot(dist_to_v0, p) * inv_det;
    if(-epsilon > u || u > 1.0f + epsilon)
        return false;

    auto q = Cross(dist_to_v0, e0);
    float v = Dot(dir, q) * inv_det;
    if(-epsilon > v || u + v > 1.0f + epsilon)
        return false;

    float t = Dot(e1, q) * inv_det;
    if(t < epsilon)
        return false;
    *out_t = t;
	*out_normal = Cross(e0, e1);
    partial_barycentric_coordinates->x = Clampf(u, 0.0f, 1.0f);
    partial_barycentric_coordinates->y = Clampf(v, 0.0f, 1.0f);
    return true;
}

#define IntersectRayTriangle IntersectRayTriangleMollerTrumbore

inline EXPORT_CUDA void TriangleBarycentricCoordinates(const Vector2& point, const Vector2& v0, const Vector2& v1, const Vector2& v2, Vector2* partial_barycentric_coordinates)
{
    auto x1_x3 = v1.x - v0.x;
    auto x2_x3 = v2.x - v0.x;
    auto x_x3 = point.x - v0.x;

    auto y1_y3 = v1.y - v0.y;
    auto y2_y3 = v2.y - v0.y;
    auto y_y3 = point.y - v0.y;    

    float det = x1_x3*y2_y3 - x2_x3*y1_y3;
    if(det < 1e-9f) // Malformed triangle
    {
        partial_barycentric_coordinates->x = partial_barycentric_coordinates->y = Length(point - v0);
        return;
    }

    partial_barycentric_coordinates->x = ( y2_y3*x_x3 - x2_x3*y_y3)/det;
    partial_barycentric_coordinates->y = (-y1_y3*x_x3 + x1_x3*y_y3)/det;
}

inline EXPORT_CUDA bool IntersectTriangle(const Vector2& point, const Vector2& v0, const Vector2& v1, const Vector2& v2, Vector3* barycentric_coordinates)
{
    Vector2 partial;
    TriangleBarycentricCoordinates(point, v0, v1, v2, &partial);

    float w = 1.0f - partial.x - partial.y;
    
    if(0.0f > partial.x || partial.x > 1.0f ||
       0.0f > partial.y || partial.y > 1.0f ||
       0.0f > w || w > 1.0f)
        return false;

    barycentric_coordinates->x = partial.x;
    barycentric_coordinates->y = partial.y;
    barycentric_coordinates->z = w;
    return true;
}

inline EXPORT_CUDA bool DistancePointLine(const Vector2& point, const Vector2& origin, const Vector2& direction, float* tmin)
{
    float tmp_dist = Dot(point - origin, direction);
    *tmin = tmp_dist;
    return true;
}

inline EXPORT_CUDA bool DistancePointLineSegment(const Vector2& point, const Vector2& origin, const Vector2& direction, const float length, float* tmin)
{
    /*Computes the location on a line segment from which the distance to a given point is minimal.
     * Requirements:
     * 0<tmp_dist<length, direction needs to be normalized.*/
    float tmp_dist = Dot(direction, point - origin);
    if (tmp_dist <= 0.0f)
    {
        *tmin = 0.0f;
        return true;
    }
    if (tmp_dist > length)
    {
        *tmin = length;
        return true;
    }
    *tmin = tmp_dist;
    return true;
}

// Super slow
bool IntersectTriangleList2D(const Vector2& point, const Vector2* vertices, uint32_t vertex_count, uint32_t* indices, uint32_t triangle_count, uint32_t* prim_id, Vector3* barycentric_coord);

template<class TVector, class TAABB>
inline EXPORT_CUDA bool IntersectPointAABB(const TVector& point, const TAABB& box)
{
	return box.MinCorner <= point && point <= box.MaxCorner;
}

inline EXPORT_CUDA bool IntersectSphereCenteredAABB(const Sphere& sphere, const Vector3& size)
{
    float dist_squared = sphere.Radius * sphere.Radius;
    /* assume C1 and C2 are element-wise sorted, if not, do that now */
    auto abs_center = Vector3Abs(sphere.Center);
    if(abs_center.x > size.x)
    {
        float value = abs_center.x - size.x;
        dist_squared -= value*value;
    }
    if(abs_center.y > size.y)
    {
        float value = abs_center.y - size.y;
        dist_squared -= value*value;
    }
    if(abs_center.z > size.z)
    {
        float value = abs_center.z - size.z;
        dist_squared -= value*value;
    }
    return dist_squared > 0;
}

inline EXPORT_CUDA bool IntersectSphereAABB(const Sphere& sphere, const AABBUnaligned& box)
{
    float dist_squared = sphere.Radius * sphere.Radius;
    /* assume C1 and C2 are element-wise sorted, if not, do that now */
    if (sphere.Center.x < box.MinCorner.x)
    {
        float value = sphere.Center.x - box.MinCorner.x;
        dist_squared -= value*value;
    }
    else if(sphere.Center.x > box.MaxCorner.x)
    {
        float value = sphere.Center.x - box.MaxCorner.x;
        dist_squared -= value*value;
    }
    if (sphere.Center.y < box.MinCorner.y)
    {
        float value = sphere.Center.y - box.MinCorner.y;
        dist_squared -= value*value;
    }
    else if (sphere.Center.y > box.MaxCorner.y) 
    {
        float value = sphere.Center.y - box.MaxCorner.y;
        dist_squared -= value*value;
    }
    if (sphere.Center.z < box.MinCorner.z)
    {
        float value = sphere.Center.z - box.MinCorner.z;
        dist_squared -= value*value;
    }
    else if (sphere.Center.z > box.MaxCorner.z)
    {
        float value = sphere.Center.z - box.MaxCorner.z;
        dist_squared -= value*value;
    }

    return dist_squared > 0;
}

template<class TAABB>
inline EXPORT_CUDA bool IntersectAABBAABB(const TAABB& lhs, const TAABB& rhs)
{
	return rhs.MinCorner <= lhs.MaxCorner &&
		   lhs.MinCorner <= rhs.MaxCorner;
}
}

#endif // _TEMPEST_INTERSECT_HH_
