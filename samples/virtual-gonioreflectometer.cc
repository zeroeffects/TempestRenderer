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

#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/math/shapes.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/patterns.hh"

#include <cstdlib>

const float FiberRadius = 0.1f;
const float FiberLength = 20.0f;
const float AngularSpeed = 0.1f;
const float YarnRadius = 1.0f;
const float FiberMinGap = 0.0f;
const float FiberSkew = -5.0f;
const uint32_t SamplesLong = 9;
const uint32_t SamplesAzimuth = 9;

int TempestMain(int argc, char** argv)
{
    float cyl_rad = YarnRadius + FiberRadius;
	float proj_horiz_span = cyl_rad;
	float proj_vert_span = proj_horiz_span;

    uint32_t image_width = 90,
             image_height = 90;

    Tempest::Matrix4 proj = Tempest::OrthoMatrix(-proj_horiz_span, proj_horiz_span, -proj_vert_span, proj_vert_span, 0.1f, 1000.0f);

    //float cur_target = -cyl_rad;

    Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                     origin{15.0f, 0.0f, 0.0f},
                     up{0.0f, 1.0f, 0.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAt(origin, target, up);

    float rot_angle_azimuth = -Tempest::MathPi;
    float rot_angle_long = -0.5f*Tempest::MathPi;
    Tempest::Matrix4 rot;
    rot.identity();
    rot.rotateX(rot_angle_long);
    rot.rotateZ(rot_angle_azimuth);

    Tempest::Matrix4 view_proj = proj * view * rot;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    Tempest::Vector3 target_in_view = view*up;

    Tempest::Matrix4 view_inv;
    view_inv = view.inverse();
    
    std::unique_ptr<Tempest::RayTracerScene> rt_scene(new Tempest::RayTracerScene(image_width, image_height, view_proj_inv));

	/*
    Tempest::SphereAreaLight* area_light = new Tempest::SphereAreaLight;
    area_light->SphereShape.Center = Tempest::Vector3{0.0f, 0.0f, 0.0f};
    area_light->SphereShape.Radius = 0.1f;
    area_light->Radiance = Tempest::Vector3{5000.0f, 5000.0f, 5000.0f};

    rt_scene->addLightSource(area_light);
	*/

	Tempest::DirectionalLight* dir_light = new Tempest::DirectionalLight;
	dir_light->Direction = Tempest::Vector3{0.0f, 1.0f, 1.0f};
    NormalizeSelf(&dir_light->Direction);
	dir_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{10.0f, 10.0f, 10.0f});
	auto light_idx = rt_scene->addLightSource(dir_light);

	rt_scene->setSamplesCamera(64);
    rt_scene->setSamplesLocalAreaLight(1);
    rt_scene->setSamplesGlobalIllumination(1);
    rt_scene->setMaxRayDepth(1);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::RGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    Tempest::RTMicrofacetMaterial material;
	memset(&material, 0x0, sizeof(material));
	
    Tempest::Spectrum color = Tempest::RGBToSpectrum(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });

    material.Model = Tempest::IlluminationModel::GGXMicrofacet;
	material.Diffuse = color*0.05f;
    material.Specular = color*0.95f;
    material.SpecularPower = Tempest::Vector2{ 10.0f, 10.0f };
	material.Fresnel.x = Tempest::ComputeReflectionCoefficient(1.56f);
    material.setup();

    struct FiberLayer
    {
        Tempest::ShapeVertex* Vertices = nullptr;
        uint32_t              VertexCount = 0;
        int32_t*              Indices = nullptr;
        uint32_t              IndexCount = 0;

        ~FiberLayer()
        {
            delete[] Vertices;
            delete[] Indices;
        }
    };

    const uint32_t stride = 6*sizeof(float);

    Tempest::Vector3 yarn_pos = target;

    float radius = (YarnRadius + FiberRadius)/3.0f;

    Tempest::Vector3 offset{ 0.0f, radius*2.0f, 0.0f };
    Tempest::Vector3 up_offset{ 2.0f*radius, 0.0f, 0.0f };

    rt_scene->addCylinder(Tempest::Cylinder{ yarn_pos - offset, radius, FiberLength }, &material);
    rt_scene->addCylinder(Tempest::Cylinder{ yarn_pos + up_offset, radius, FiberLength }, &material);
    rt_scene->addCylinder(Tempest::Cylinder{ yarn_pos + offset, radius, FiberLength }, &material);

	rt_scene->commitScene();

    Tempest::TimeQuery query;
    auto start = query.time();

    rt_scene->initWorkers();

    size_t tex_size = SamplesAzimuth*SamplesLong*image_width*image_height;
    auto final_image = std::unique_ptr<uint32_t[]>(new uint32_t[tex_size]);

    std::fill(final_image.get(), final_image.get() + tex_size, 0x00FF00FF);

    uint32_t image_size = image_width*image_height;
    
    //rt_scene->setRenderMode(Tempest::RenderMode::DebugNormals);

    for(uint32_t samp = 0, samp_end = SamplesLong*SamplesAzimuth; samp < samp_end; ++samp)
    {
        uint32_t next_samp = samp + 1;
        uint32_t next_samp_azimuth = next_samp % SamplesAzimuth;
        uint32_t next_samp_long = next_samp / SamplesAzimuth;
        uint32_t samp_azimuth = samp % SamplesAzimuth;
        uint32_t samp_long = samp / SamplesAzimuth;

    //    Tempest::Log(Tempest::LogLevel::Debug, rot_angle_azimuth, "(", samp_azimuth, "), ", rot_angle_long, "(", samp_long, ")");
        Tempest::Vector2 tc{ 2.0f*next_samp_azimuth/(SamplesAzimuth - 1) - 1.0f, 2.0f*next_samp_long/(SamplesLong - 1) - 1.0f };

        float sin_theta2 = Tempest::Clampf(Tempest::Dot(tc, tc), 0.0f, 1.0f);
        float cos_theta = sqrtf(1.0f - sin_theta2);
        float sin_theta = sqrtf(sin_theta2);

        Tempest::Log(Tempest::LogLevel::Info, "Finished: ", samp_long, ":", samp_azimuth);

        auto* frame_data = rt_scene->drawOnce();

        dir_light->Direction = Tempest::Vector3{-cos_theta, -sin_theta*tc.x, -sin_theta*tc.y};
        rt_scene->updateLightSource(light_idx, dir_light);

        for(uint32_t row = 0; row < image_height; ++row)
        {
            std::copy_n(reinterpret_cast<uint32_t*>(frame_data->Backbuffer->getData()) + image_width * row, image_width, final_image.get() + image_width*((samp_long*image_height + row)*SamplesAzimuth + samp_azimuth));
        }

        rt_scene->draw(image_width, image_height, view_proj_inv);
    }

    auto end = query.time();

    Tempest::Log(Tempest::LogLevel::Info, "Render time (ray tracing): ", end - start, "us");
   
    Tempest::TextureDescription tex_desc;
    tex_desc.Width = SamplesAzimuth*image_width;
    tex_desc.Height = SamplesLong*image_height;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;
    
    Tempest::SaveImage(tex_desc, final_image.get(), Tempest::Path("yarn.tga"));

    Tempest::DisplayImage(tex_desc, final_image.get());
    return EXIT_FAILURE;
}