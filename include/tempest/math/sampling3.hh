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

#ifndef _TEMPEST_SAMPLING3_HH_
#define _TEMPEST_SAMPLING3_HH_

#include "tempest/math/vector3.hh"
#include "tempest/math/matrix3.hh"
#include "tempest/math/triangle.hh"

namespace Tempest
{
inline EXPORT_CUDA float VonMisesFisherPDF(const Tempest::Vector3& mean_dir, float concentration, const Tempest::Vector3& dir)
{
    if(concentration == 0.0f)
        return 1.0f/(4.0f*Tempest::MathPi);

    float coef = concentration/(2.0f*Tempest::MathPi*(1.0f - expf(-2.0f*concentration)));
    return coef*expf(concentration*(Tempest::Dot(mean_dir, dir) - 1.0f));
}

inline EXPORT_CUDA float VonMisesFisherConeCDF(float concentration, float cos_angle)
{
    return (1.0f - expf(concentration*(cos_angle - 1.0f)))/(1.0f - expf(-2.0f*concentration));
}

inline EXPORT_CUDA float VonMisesFisherToleranceCone(float concentration, float required_tolerance)
{
    TGE_ASSERT(concentration != 0.0f, "TODO: replace with cone area");

    return (logf(1.0f - required_tolerance*(1.0f - expf(-2.0f*concentration))))/concentration + 1.0f;
}

inline EXPORT_CUDA Vector3 UniformSampleHemisphere(float ra, float rb)
{
    float r = sqrtf(1.0f - ra*ra);
    float phi = 2.0f*MathPi*rb;
    float sin_phi, cos_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);
    return Vector3{cos_phi * r, sin_phi * r, ra};
}

inline EXPORT_CUDA Vector2 UniformSampleHemisphereSeed(const Vector3& sample)
{
    return { sample.z, atan2f(sample.y, sample.x)/(2.0f*MathPi) };
}

inline EXPORT_CUDA Vector3 CosineSampleHemisphere(float ra, float rb)
{
    float r = sqrtf(ra);
    float phi = 2.0f*MathPi*rb;
    float sin_phi, cos_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);
    return Vector3{cos_phi * r, sin_phi * r, sqrtf(Maxf(0.0f, 1.0f - ra))};
}

inline EXPORT_CUDA Vector3 AnisoPowerCosineSampleHemisphere(float ra, float rb, const Vector2& power)
{
	// NOTE: fails on 1.0f
	float angle = fmod(rb, 0.25f) / 0.25f;
	float cmp = rb >= 0.5f;
	angle = cmp + (1.0f - 2.0f * cmp) * angle;
	angle *= MathPi*0.5f;
	float phi = atanf(sqrtf((power.x + 1.0f)/(power.y + 1.0f))*tanf(angle));
	
	float cos_phi = cosf(phi);
    TGE_ASSERT(phi <= MathPi, "Sign cannot be ignored");
	float sin_phi = sqrtf(1.0f - cos_phi*cos_phi);
	float cos_theta = powf(1.0f - ra, 1.0f / (power.x*cos_phi*cos_phi + power.y*sin_phi*sin_phi + 1.0f));
	float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
	
	return Vector3{cos_phi * sin_theta, sin_phi * sin_theta, cos_theta };
}

inline EXPORT_CUDA Vector3 PowerCosineSampleHemisphere(float ra, float rb, float power)
{
    float cos_theta = powf(ra, 1.0f/(power + 1.0f));
    float sin_theta = sqrtf(Maxf(0.0f, 1.0f - cos_theta*cos_theta));
    float phi = 2.0f*MathPi*rb;
    float cos_phi, sin_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);
    return Vector3{cos_phi * sin_theta, sin_phi * sin_theta, cos_theta};
}

inline EXPORT_CUDA Vector3 UniformSampleSphere(float ra, float rb)
{
    float phi = 2*MathPi*ra;
    float sin_phi, cos_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);
    float cos_theta = 2.0f*rb - 1.0f;
    float sin_theta = sqrtf(1 - cos_theta*cos_theta);
    Vector3 ret{cos_phi*sin_theta, sin_phi*sin_theta, cos_theta};
    TGE_ASSERT(Tempest::ApproxEqual(Tempest::Length(ret), 1.0f), "Bad uniform sphere sample generator");
    return ret;
}

inline EXPORT_CUDA float UniformSphericalConePDF(float cos_angle)
{
    return 1.0f/(2.0f*MathPi*(1.0f - cos_angle));
}

inline EXPORT_CUDA Vector3 UniformSampleSphericalCone(float cos_angle, float ra, float rb)
{
    float phi = 2*MathPi*ra;
    float sin_phi, cos_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);
    float cos_theta = 1.0f + rb*(cos_angle - 1.0f);
    float sin_theta = sqrtf(1 - cos_theta*cos_theta);
    Vector3 ret{cos_phi*sin_theta, sin_phi*sin_theta, cos_theta};
    TGE_ASSERT(Tempest::ApproxEqual(Tempest::Length(ret), 1.0f), "Bad uniform sphere sample generator");
    return ret;
}

struct SampleIntersect
{
    Vector3 Direction;
    float   Distance;
    float   PDF;
};

inline EXPORT_CUDA SampleIntersect UniformSampleSphereArea(const Vector3& pos, const Vector3& center, float radius, float ra, float rb)
{
    float phi = 2.0f*MathPi*rb;
    float _2radius = 2.0f*radius;
    float sin_phi, cos_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);
    Vector3 result = center - pos + Vector3{_2radius*cos_phi*sqrtf(ra*(1.0f - ra)),
                                            _2radius*sin_phi*sqrtf(ra*(1.0f - ra)),
                                            radius*(1 - 2*ra)};
    float t = Length(result);

    return SampleIntersect{ result / t, t, 1.0f/(4.0f*MathPi) };
}

// Shirley, et al
inline EXPORT_CUDA SampleIntersect SampleSphereArea(const Vector3& pos, const Vector3& center, float radius, float ra, float rb)
{
    Vector3 dist_to_center = center - pos;
    float dist2 = Dot(dist_to_center, dist_to_center);
    float dist = sqrtf(dist2);
    if(dist <= radius)
    {
        return UniformSampleSphereArea(pos, center, radius, ra, rb);
    }

    Vector3 dir_to_center = dist_to_center;
    dir_to_center /= dist;

    float _sin_rc = radius / dist;
    float _cos_rc = sqrtf(1 - _sin_rc*_sin_rc);

    float phi = 2*MathPi*ra;
    float sin_phi, cos_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);
    float cos_theta = 1.0f - rb + rb*_cos_rc;
    float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);

    Matrix3 surface_space;
    surface_space.makeBasis(dir_to_center);
    
    Vector3 smp_dir{cos_phi*sin_theta, sin_phi*sin_theta, cos_theta};
    auto dir = surface_space.transform(smp_dir);
    NormalizeSelf(&dir);

    float b = dist / cos_theta; //dir.dot(dist_to_center);
	float c = dist2 - radius*radius;

	TGE_ASSERT(b >= 0.0f, "Invalid computation");

    float D = b*b - c /* * dir.dot(dir) */;

    float t = b;

    if(D >= 0.0f)
    {
        t -= sqrtf(D);
    }

    float pdf = 1.0f / (2.0f*MathPi * (1.0f - _cos_rc)); // distance omitted

    return SampleIntersect{ dir, t, pdf };
}

inline EXPORT_CUDA float ProjectedTriangleSphereAreaApproximate(const Vector3& pos, const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
    Tempest::Vector3 p0_proj = Normalize(p0 - pos),
                     p1_proj = Normalize(p1 - pos),
                     p2_proj = Normalize(p2 - pos);

    return TriangleArea(p0_proj, p1_proj, p2_proj);
}

inline EXPORT_CUDA SampleIntersect UniformSampleTriangleArea(const Vector3& pos, const Vector3& p0, const Vector3& p1, const Vector3& p2, float ra, float rb)
{
    SampleIntersect sample_intersect;
    float sqrt_r = sqrtf(ra); // 1 - ra
    auto e0 = p1 - p0,
         e1 = p2 - p0;
    auto point = p0 + sqrt_r*rb*e0 + (1 - sqrt_r)*e1;
    auto dir_unorm = point - pos;
    auto len = Length(dir_unorm);
    sample_intersect.Direction = dir_unorm/len; 
    sample_intersect.Distance = len;
    sample_intersect.PDF = Dot(Cross(e0, e1), sample_intersect.Direction) < -TEMPEST_WEAK_FLOAT_EPSILON ? 1.0f/ProjectedTriangleSphereAreaApproximate(pos, p0, p1, p2) : 0.0f;
    return sample_intersect;
}

inline EXPORT_CUDA float ComputeSphereAreaPDF(const Vector3& pos, const Vector3& dir, const Vector3& center, float radius)
{
    Vector3 dist_to_center = center - pos;
    float dist2 = Dot(dist_to_center, dist_to_center);
    float dist = sqrtf(dist2);
    if(dist <= radius)
    {
        return 1.0f/(4.0f*MathPi);
    }

    Vector3 dir_to_center = dist_to_center;
    dir_to_center /= dist;

    float _sin_rc = radius / dist;
    float _cos_rc = sqrtf(1 - _sin_rc*_sin_rc);

    float cos_theta = Dot(dir, dir_to_center);
    if(cos_theta < _cos_rc)
        return 0.0f;
    /*
    float b = dist / cos_theta;
	float c = dist2 - radius*radius;

	TGE_ASSERT(b >= 0.0f, "Invalid computation");

    float D = b*b - c;

    float t = b;

    if(D >= 0.0f)
    {
        t -= sqrtf(D);
    }
    //*/
    return 1.0f / (2.0f*MathPi*(1.0f - _cos_rc));
}
}

#endif // _TEMPEST_SAMPLING3_HH_
