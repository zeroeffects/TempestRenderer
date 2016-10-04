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

#ifndef _TEMPEST_SHAPES_HH_
#define _TEMPEST_SHAPES_HH_

#include "tempest/math/matrix3.hh"
#include "tempest/math/shapes.hh"
#include "tempest/utils/logging.hh"

namespace Tempest
{
void TriangleTessellationNoNormals(const Sphere& sphere, uint32_t sphere_long_tes, uint32_t sphere_lat_tes, Vector3** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count)
{
    auto regular_grid_size = (sphere_lat_tes - 1)*sphere_long_tes*6;
    *indices_count = regular_grid_size + sphere_long_tes*6;
    *indices = new int32_t[*indices_count];

    *vert_count = sphere_lat_tes*sphere_long_tes + 2;
    *verts = new Vector3[*vert_count];
    
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

            auto& vert = (*verts)[vert_idx++];

            Tempest::Vector3 norm = { cos_phi*sin_theta, sin_phi*sin_theta, cos_theta };

            vert = sphere.Center + norm*sphere.Radius;

            if(latitude != sphere_lat_tes - 1)
            {
                int32_t next_vertex = longitude != sphere_long_tes - 1 ? 1 : -(int32_t)(sphere_long_tes - 1);

                uint32_t base_idx = latitude*sphere_long_tes + longitude;
                (*indices)[idx++] = base_idx;
                (*indices)[idx++] = base_idx + next_vertex + sphere_long_tes;
                (*indices)[idx++] = base_idx + next_vertex;
                (*indices)[idx++] = base_idx;
                (*indices)[idx++] = base_idx + sphere_long_tes;
                (*indices)[idx++] = base_idx + next_vertex + sphere_long_tes;
            }
        }
    }

    {
    auto& vert = (*verts)[vert_idx++];
    Tempest::Vector3 norm = { 0.0f, 0.0f,  1.0f };
    vert = sphere.Center + norm*sphere.Radius;
    }
    
    {
    auto& vert = (*verts)[vert_idx++];
    Tempest::Vector3 norm = { 0.0f, 0.0f, -1.0f };
    vert = sphere.Center + norm*sphere.Radius;
    }

    TGE_ASSERT(idx == regular_grid_size, "Invalid index count of the main grid");
    TGE_ASSERT(vert_idx == *vert_count, "Invalid vertex population");

    for(uint32_t longitude = 0; longitude < sphere_long_tes; ++longitude)
    {
        int32_t next_vertex = longitude != sphere_long_tes - 1 ? 1 : -(int32_t)(sphere_long_tes - 1);

        (*indices)[idx++] = sphere_lat_tes*sphere_long_tes;
        (*indices)[idx++] = longitude;
        (*indices)[idx++] = longitude + next_vertex;

        (*indices)[idx++] = sphere_lat_tes*sphere_long_tes + 1;
        uint32_t base_idx = (sphere_lat_tes - 1)*sphere_long_tes;
        (*indices)[idx++] = base_idx + longitude;
        (*indices)[idx++] = base_idx + longitude + next_vertex;
    }

    TGE_ASSERT(idx == *indices_count, "Invalid index count");

#ifndef NDEBUG
    for(uint32_t idx = 0; idx < *indices_count; ++idx)
    {
        TGE_ASSERT((*indices)[idx] < *vert_count, "Invalid normal length");
    }
#endif
}


void TriangleTessellation(const Sphere& sphere, uint32_t sphere_long_tes, uint32_t sphere_lat_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count)
{
    auto regular_grid_size = (sphere_lat_tes - 1)*sphere_long_tes*6;
    *indices_count = regular_grid_size + sphere_long_tes*6;
    *indices = new int32_t[*indices_count];

    *vert_count = sphere_lat_tes*sphere_long_tes + 2;
    *verts = new ShapeVertex[*vert_count];
    
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

            auto& vert = (*verts)[vert_idx++];

            Tempest::Vector3 norm = { cos_phi*sin_theta, sin_phi*sin_theta, cos_theta };

            vert.Position = sphere.Center + norm*sphere.Radius;
            vert.Normal = norm;

            if(latitude != sphere_lat_tes - 1)
            {
                int32_t next_vertex = longitude != sphere_long_tes - 1 ? 1 : -(int32_t)(sphere_long_tes - 1);

                uint32_t base_idx = latitude*sphere_long_tes + longitude;
                (*indices)[idx++] = base_idx;
                (*indices)[idx++] = base_idx + next_vertex + sphere_long_tes;
                (*indices)[idx++] = base_idx + next_vertex;
                (*indices)[idx++] = base_idx;
                (*indices)[idx++] = base_idx + sphere_long_tes;
                (*indices)[idx++] = base_idx + next_vertex + sphere_long_tes;
            }
        }
    }

    {
    auto& vert = (*verts)[vert_idx++];
    Tempest::Vector3 norm = { 0.0f, 0.0f,  1.0f };
    vert.Position = sphere.Center + norm*sphere.Radius;
    vert.Normal = norm;
    }
    
    {
    auto& vert = (*verts)[vert_idx++];
    Tempest::Vector3 norm = { 0.0f, 0.0f, -1.0f };
    vert.Position = sphere.Center + norm*sphere.Radius;
    vert.Normal = norm;    
    }

    TGE_ASSERT(idx == regular_grid_size, "Invalid index count of the main grid");
    TGE_ASSERT(vert_idx == *vert_count, "Invalid vertex population");

    for(uint32_t longitude = 0; longitude < sphere_long_tes; ++longitude)
    {
        int32_t next_vertex = longitude != sphere_long_tes - 1 ? 1 : -(int32_t)(sphere_long_tes - 1);

        (*indices)[idx++] = sphere_lat_tes*sphere_long_tes;
        (*indices)[idx++] = longitude;
        (*indices)[idx++] = longitude + next_vertex;

        (*indices)[idx++] = sphere_lat_tes*sphere_long_tes + 1;
        uint32_t base_idx = (sphere_lat_tes - 1)*sphere_long_tes;
        (*indices)[idx++] = base_idx + longitude;
        (*indices)[idx++] = base_idx + longitude + next_vertex;
    }

    TGE_ASSERT(idx == *indices_count, "Invalid index count");

#ifndef NDEBUG
    for(uint32_t idx = 0; idx < *indices_count; ++idx)
    {
        TGE_ASSERT((*indices)[idx] < *vert_count, "Invalid normal length");
    }
    
    for(uint32_t vert_idx = 0; vert_idx < *vert_count; ++vert_idx)
    {
        TGE_ASSERT(Tempest::ApproxEqual(Tempest::Length((*verts)[vert_idx].Normal), 1.0f, 1e-3f), "Invalid normal length");
    }
#endif
}

void TriangleTessellation(const Cylinder& cylinder, uint32_t circular_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count)
{
    TGE_ASSERT(circular_tes > 2, "Invalid tesselation");

    *vert_count = 4*circular_tes;
    *verts = new ShapeVertex[*vert_count];

    *indices_count = (circular_tes - 2)*3*2 + circular_tes*6;
    *indices = new int32_t[*indices_count];

    uint32_t vert_idx = 0, ind_idx = 0;
    float theta = 0.0f;
    float cos_theta = 1.0f, sin_theta = 0.0f;

    const uint32_t wrap_point = circular_tes*4;

    for(uint32_t i = 0; i < circular_tes; ++i)
    {
        float x = cylinder.Radius*cos_theta;
        float y = cylinder.Radius*sin_theta;

        if(0 < i && i < circular_tes - 1)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + 4) % wrap_point;
            (*indices)[ind_idx++] = 0;
        }

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x, y, cylinder.HalfHeight } + cylinder.Center,
                                            Vector3{ 0.0f, 0.0f, 1.0f } };

        (*indices)[ind_idx++] = vert_idx;
        (*indices)[ind_idx++] = (vert_idx + 1);
        (*indices)[ind_idx++] = (vert_idx + 1 + 4) % wrap_point;

        (*indices)[ind_idx++] = vert_idx;
        (*indices)[ind_idx++] = (vert_idx + 1 + 4) % wrap_point;
        (*indices)[ind_idx++] = (vert_idx + 4) % wrap_point;

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x, y, cylinder.HalfHeight } + cylinder.Center,
                                            Vector3{ cos_theta, sin_theta, 0.0f } };

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x, y, -cylinder.HalfHeight } + cylinder.Center,
                                            Vector3{ cos_theta, sin_theta, 0.0f } };

        // yeah, i know
        if(0 < i && i < circular_tes - 1)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + 4) % wrap_point;
            (*indices)[ind_idx++] = 3;
        }

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x, y, -cylinder.HalfHeight } + cylinder.Center,
                                            Vector3{ 0.0f, 0.0f, -1.0f } };
        

        float theta = 2.0f*MathPi*(i + 1)/circular_tes;
        FastSinCos(theta, &sin_theta, &cos_theta);
    }

    #ifndef NDEBUG
    TGE_ASSERT(vert_idx == *vert_count && ind_idx == *indices_count, "Invalid vertices or indices");
    for(uint32_t i = 0; i < ind_idx; ++i)
    {
        TGE_ASSERT((*indices)[i] < (int32_t)vert_idx, "Invalid index");
    }

    TGE_ASSERT((ind_idx % 3) == 0, "Invalid index count");
    #endif
}

void TriangleTessellation(const Cylinder& cylinder, uint32_t circular_tes, uint32_t vert_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count)
{
    TGE_ASSERT(circular_tes > 2 && vert_tes > 1, "Invalid tesselation");

    uint32_t col_size = 2 + vert_tes;
    *vert_count = col_size*circular_tes;
    *verts = new ShapeVertex[*vert_count];

    *indices_count = (circular_tes - 2)*3*2 + circular_tes*(vert_tes - 1)*6;
    *indices = new int32_t[*indices_count];

    uint32_t vert_idx = 0, ind_idx = 0;
    float theta = 0.0f;
    float cos_theta = 1.0f, sin_theta = 0.0f;

    const uint32_t wrap_point = circular_tes*col_size;

    for(uint32_t i = 0; i < circular_tes; ++i)
    {
        float x = cylinder.Radius*cos_theta;
        float y = cylinder.Radius*sin_theta;

        if(0 < i && i < circular_tes - 1)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + col_size) % wrap_point;
            (*indices)[ind_idx++] = 0;
        }

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x, y, cylinder.HalfHeight } + cylinder.Center,
                                            Vector3{ 0.0f, 0.0f, 1.0f } };

        for(uint32_t j = 0; j < vert_tes - 1; ++j)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + 1);
            (*indices)[ind_idx++] = (vert_idx + 1 + col_size) % wrap_point;

            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + 1 + col_size) % wrap_point;
            (*indices)[ind_idx++] = (vert_idx + col_size) % wrap_point;

            float cur_height = cylinder.HalfHeight - 2.0f*cylinder.HalfHeight*j/(vert_tes - 1);
            Vector3 pos = Vector3{ x, y, cur_height };

            Vector3 norm = Vector3{ pos.x, pos.y, 0.0f }; // *2.0f Omitted
            NormalizeSelf(&norm);
            (*verts)[vert_idx++] = ShapeVertex{ pos + cylinder.Center,
                                                norm };
        }

        // Make sure that everything is water-tight
        Vector3 pos = Vector3{ x, y, -cylinder.HalfHeight };
        Vector3 norm = Vector3{ pos.x, pos.y, 0.0f }; // *2.0f Omitted
        NormalizeSelf(&norm);
        (*verts)[vert_idx++] = ShapeVertex{ pos + cylinder.Center,
                                            norm };

        // yeah, i know
        if(0 < i && i < circular_tes - 1)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + col_size) % wrap_point;
            (*indices)[ind_idx++] = col_size - 1;
        }

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x, y, -cylinder.HalfHeight } + cylinder.Center,
                                            Vector3{ 0.0f, 0.0f, -1.0f } };
        

        float theta = 2.0f*MathPi*(i + 1)/circular_tes;
        FastSinCos(theta, &sin_theta, &cos_theta);
    }

    #ifndef NDEBUG
    TGE_ASSERT(vert_idx == *vert_count && ind_idx == *indices_count, "Invalid vertices or indices");
    for(uint32_t i = 0; i < ind_idx; ++i)
    {
        TGE_ASSERT((*indices)[i] < (int32_t)vert_idx, "Invalid index");
    }

    TGE_ASSERT((ind_idx % 3) == 0, "Invalid index count");
    #endif
}

void TriangleTessellation(const ObliqueCylinder& cylinder, uint32_t circular_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count)
{
    TGE_ASSERT(circular_tes > 2, "Invalid tesselation");

    *vert_count = 4*circular_tes;
    *verts = new ShapeVertex[*vert_count];

    *indices_count = (circular_tes - 2)*3*2 + circular_tes*6;
    *indices = new int32_t[*indices_count];

    uint32_t vert_idx = 0, ind_idx = 0;
    float theta = 0.0f;
    float cos_theta = 1.0f, sin_theta = 0.0f;

    const uint32_t wrap_point = circular_tes*4;

    for(uint32_t i = 0; i < circular_tes; ++i)
    {
        float x = cylinder.CylinderShape.Radius*cos_theta;
        float y = cylinder.CylinderShape.Radius*sin_theta;

        if(0 < i && i < circular_tes - 1)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + 4) % wrap_point;
            (*indices)[ind_idx++] = 0;
        }

        Tempest::Vector2 bending = cylinder.TiltDirection*cylinder.Tilt;

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x + bending.x, y + bending.y, cylinder.CylinderShape.HalfHeight } + cylinder.CylinderShape.Center,
                                            Vector3{ 0.0f, 0.0f, 1.0f } };

        (*indices)[ind_idx++] = vert_idx;
        (*indices)[ind_idx++] = (vert_idx + 1);
        (*indices)[ind_idx++] = (vert_idx + 1 + 4) % wrap_point;

        (*indices)[ind_idx++] = vert_idx;
        (*indices)[ind_idx++] = (vert_idx + 1 + 4) % wrap_point;
        (*indices)[ind_idx++] = (vert_idx + 4) % wrap_point;

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x + bending.x, y + bending.y, cylinder.CylinderShape.HalfHeight } + cylinder.CylinderShape.Center,
                                            Vector3{ cos_theta, sin_theta, 0.0f } };

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x - bending.x, y - bending.y, -cylinder.CylinderShape.HalfHeight } + cylinder.CylinderShape.Center,
                                            Vector3{ cos_theta, sin_theta, 0.0f } };

        // yeah, i know
        if(0 < i && i < circular_tes - 1)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + 4) % wrap_point;
            (*indices)[ind_idx++] = 3;
        }

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x - bending.x, y - bending.y, -cylinder.CylinderShape.HalfHeight } + cylinder.CylinderShape.Center,
                                            Vector3{ 0.0f, 0.0f, -1.0f } };
        

        float theta = 2.0f*MathPi*(i + 1)/circular_tes;
        FastSinCos(theta, &sin_theta, &cos_theta);
    }

    #ifndef NDEBUG
    TGE_ASSERT(vert_idx == *vert_count && ind_idx == *indices_count, "Invalid vertices or indices");
    for(uint32_t i = 0; i < ind_idx; ++i)
    {
        TGE_ASSERT((*indices)[i] < (int32_t)vert_idx, "Invalid index");
    }

    TGE_ASSERT((ind_idx % 3) == 0, "Invalid index count");
    #endif
}

void TriangleTessellation(const HelixCylinder& cylinder, uint32_t circular_tes, uint32_t vert_tes, ShapeVertex** verts, uint32_t* vert_count, int32_t** indices, uint32_t* indices_count)
{
    TGE_ASSERT(circular_tes > 2 && vert_tes > 1, "Invalid tesselation");

    uint32_t col_size = 2 + vert_tes;
    *vert_count = col_size*circular_tes;
    *verts = new ShapeVertex[*vert_count];

    *indices_count = (circular_tes - 2)*3*2 + circular_tes*(vert_tes - 1)*6;
    *indices = new int32_t[*indices_count];

    uint32_t vert_idx = 0, ind_idx = 0;
    float theta = 0.0f;
    float cos_theta = 1.0f, sin_theta = 0.0f;

    const uint32_t wrap_point = circular_tes*col_size;

    for(uint32_t i = 0; i < circular_tes; ++i)
    {
        float x = cylinder.CylinderShape.Radius*cos_theta;
        float y = cylinder.CylinderShape.Radius*sin_theta;

        if(0 < i && i < circular_tes - 1)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + col_size) % wrap_point;
            (*indices)[ind_idx++] = 0;
        }

        float displacement_angle = MathPi*cylinder.CylinderShape.HalfHeight*cylinder.AngularSpeed;
        float sin_displacement_angle, cos_displacement_angle;
        FastSinCos(displacement_angle, &sin_displacement_angle, &cos_displacement_angle);
        float disp_x = cylinder.CurvatureRadius*cos_displacement_angle;
        float disp_y = cylinder.CurvatureRadius*sin_displacement_angle;

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x + disp_x, y + disp_y, cylinder.CylinderShape.HalfHeight } + cylinder.CylinderShape.Center,
                                            Vector3{ 0.0f, 0.0f, 1.0f } };

        for(uint32_t j = 0; j < vert_tes - 1; ++j)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + 1);
            (*indices)[ind_idx++] = (vert_idx + 1 + col_size) % wrap_point;

            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + 1 + col_size) % wrap_point;
            (*indices)[ind_idx++] = (vert_idx + col_size) % wrap_point;

            float cur_height = cylinder.CylinderShape.HalfHeight - 2.0f*cylinder.CylinderShape.HalfHeight*j/(vert_tes - 1);
            displacement_angle = MathPi*cur_height*cylinder.AngularSpeed;
            FastSinCos(displacement_angle, &sin_displacement_angle, &cos_displacement_angle);
            disp_x = cylinder.CurvatureRadius*cos_displacement_angle;
            disp_y = cylinder.CurvatureRadius*sin_displacement_angle;
            Vector3 pos = Vector3{ x + disp_x, y + disp_y, cur_height };

            TGE_ASSERT((pos.x - disp_x)*(pos.x - disp_x) + (pos.y - disp_y)*(pos.y - disp_y) - cylinder.CylinderShape.Radius*cylinder.CylinderShape.Radius < 1e-3f, "");

            Vector3 norm = Vector3{ pos.x - disp_x, pos.y - disp_y, MathPi * cylinder.AngularSpeed * (pos.x * disp_y - pos.y * disp_x) }; // *2.0f Omitted
            NormalizeSelf(&norm);
            (*verts)[vert_idx++] = ShapeVertex{ pos + cylinder.CylinderShape.Center,
                                                norm };
        }

        // Make sure that everything is water-tight
        displacement_angle = -MathPi*cylinder.CylinderShape.HalfHeight*cylinder.AngularSpeed;
        FastSinCos(displacement_angle, &sin_displacement_angle, &cos_displacement_angle);
        disp_x = cylinder.CurvatureRadius*cos_displacement_angle;
        disp_y = cylinder.CurvatureRadius*sin_displacement_angle;
        Vector3 pos = Vector3{ x + disp_x, y + disp_y, -cylinder.CylinderShape.HalfHeight };
        Vector3 norm = Vector3{ pos.x - disp_x, pos.y - disp_y, MathPi * cylinder.AngularSpeed * (pos.x * disp_y - pos.y * disp_x) }; // *2.0f Omitted
        NormalizeSelf(&norm);
        (*verts)[vert_idx++] = ShapeVertex{ pos + cylinder.CylinderShape.Center,
                                            norm };

        // yeah, i know
        if(0 < i && i < circular_tes - 1)
        {
            (*indices)[ind_idx++] = vert_idx;
            (*indices)[ind_idx++] = (vert_idx + col_size) % wrap_point;
            (*indices)[ind_idx++] = col_size - 1;
        }

        (*verts)[vert_idx++] = ShapeVertex{ Vector3{ x + disp_x, y + disp_y, -cylinder.CylinderShape.HalfHeight } + cylinder.CylinderShape.Center,
                                            Vector3{ 0.0f, 0.0f, -1.0f } };
        

        float theta = 2.0f*MathPi*(i + 1)/circular_tes;
        FastSinCos(theta, &sin_theta, &cos_theta);
    }

    #ifndef NDEBUG
    TGE_ASSERT(vert_idx == *vert_count && ind_idx == *indices_count, "Invalid vertices or indices");
    for(uint32_t i = 0; i < ind_idx; ++i)
    {
        TGE_ASSERT((*indices)[i] < (int32_t)vert_idx, "Invalid index");
    }

    TGE_ASSERT((ind_idx % 3) == 0, "Invalid index count");

    //*
    for(uint32_t i = 0; i < ind_idx;)
    {
        auto v0 = (*verts)[(*indices)[i++]];
        auto v1 = (*verts)[(*indices)[i++]];
        auto v2 = (*verts)[(*indices)[i++]];
        auto norm0 = v0.Normal;
        auto norm1 = v1.Normal;
        auto norm2 = v2.Normal;
        Vector3 face_norm = Cross(v1.Position - v0.Position, v2.Position - v0.Position);
        NormalizeSelf(&face_norm);
        float cor_norm = 3.0f - (fabsf(Dot(norm0, norm1)) + fabsf(Dot(norm0, norm2)) + fabsf(Dot(norm1, norm2)));
        float cor_face = 1.0f - fabsf(Dot(face_norm, norm0));
        TGE_ASSERT(cor_face <= cor_norm + 1e-5f, "Invalid normal");
    }
    //*/
    #endif
}

void CurveSkewYZ(const ShapeVertex* in_verts, uint32_t vert_count, float magnitude, float max_half_len, ShapeVertex* out_verts)
{
    float phase_fact = MathPi*0.5f/max_half_len;
    for(uint32_t vid = 0; vid < vert_count; ++vid)
    {
        auto& in_vert = in_verts[vid];
        auto& out_vert = out_verts[vid];
        TGE_ASSERT(fabsf(in_vert.Position.z) < max_half_len + 1e-3f, "Invalid input");
        float theta = phase_fact*in_vert.Position.z;
        float sin_theta, cos_theta;
        FastSinCos(theta, &sin_theta, &cos_theta);
        float offset = magnitude*cos_theta;
        out_vert.Position = Vector3{in_vert.Position.x, in_vert.Position.y + offset, in_vert.Position.z};

        Tempest::Vector3 norm{ 0.0f, 1.0f, phase_fact*magnitude*sin_theta };
        NormalizeSelf(&norm);
        Tempest::Vector3 binorm{ 1.0f, 0.0f, 0.0f };
        Tempest::Vector3 tangent{ 0.0f, -norm.z, norm.y };

        Tempest::Matrix3 mat(binorm, norm, tangent);

        out_vert.Normal = mat.transform(in_vert.Normal);
    }
}
}

#endif // _TEMPEST_SHAPES_HH_