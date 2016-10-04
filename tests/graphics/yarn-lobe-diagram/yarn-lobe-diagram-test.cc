#include "tempest/utils/testing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/math/shapes.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/patterns.hh"

const float FiberRadius = 0.05f;
const float FiberLength = 20.0f;
const float AngularSpeed = 0.1f;
const float YarnRadius = 1.0f;
const float FiberMinGap = 0.0f;
const float FiberSkew = -5.0f;
const uint32_t SamplesLong = 9;
const uint32_t SamplesAzimuth = 9;

TGE_TEST("Establishing ground truth on cylinder rendering")
{
    float cyl_rad = YarnRadius + FiberRadius;
	float proj_horiz_span = cyl_rad;
	float proj_vert_span = proj_horiz_span;

    uint32_t image_width = 100,
             image_height = 100;

    Tempest::Matrix4 proj = Tempest::OrthoMatrix(-proj_horiz_span, proj_horiz_span, -proj_vert_span, proj_vert_span, 0.1f, 1000.0f);

    //float cur_target = -cyl_rad;

    Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                     origin{0.0f, -15.0f, 0.0f},
                     up{0.0f, 0.0f, 1.0f};

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

	float dir_light_angle = 0.0f;
	float sin_theta, cos_theta;
    Tempest::FastSinCos(Tempest::ToRadians(dir_light_angle), &sin_theta, &cos_theta);

    float dir_light_alt_angle = 0.0f;
    float sin_phi, cos_phi;
    Tempest::FastSinCos(Tempest::ToRadians(dir_light_alt_angle), &sin_phi, &cos_phi);

	Tempest::DirectionalLight* dir_light = new Tempest::DirectionalLight;
	dir_light->Direction = Tempest::Vector3{-sin_theta*cos_phi, -cos_theta*cos_phi, sin_phi};
    NormalizeSelf(&dir_light->Direction);
	dir_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{10.0f, 10.0f, 10.0f});
	rt_scene->addLightSource(dir_light);

	rt_scene->setSamplesCamera(64);
    rt_scene->setSamplesLocalAreaLight(1);
    rt_scene->setSamplesGlobalIllumination(1);
    rt_scene->setMaxRayDepth(6);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::RGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    Tempest::RTMicrofacetMaterial material;
	memset(&material, 0x0, sizeof(material));
	
    Tempest::Spectrum color = Tempest::RGBToSpectrum(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });

    material.Model = Tempest::IlluminationModel::GGXMicrofacet;
	material.Diffuse = color*0.0f;
    material.Specular = color*1.0f;
    material.SpecularPower = Tempest::Vector2{ 100.0f, 100.0f };
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
//    yarn_pos.x -= cyl_rad;

    //*
    uint32_t layer_count = (uint32_t)(YarnRadius/(2.0f*FiberRadius + FiberMinGap));
    std::unique_ptr<FiberLayer[]> layers(new FiberLayer[layer_count]);

    for(uint32_t layer_idx = 0; layer_idx < layer_count; ++layer_idx)
    {
        auto& layer = layers[layer_idx];

        const float cur_radius = layer_count > 1 ? ((float)layer_idx/(layer_count - 1))*YarnRadius : YarnRadius;

        Tempest::TriangleTessellation(Tempest::HelixCylinder{ Tempest::Cylinder{ Tempest::Vector3{}, FiberRadius, FiberLength }, AngularSpeed, cur_radius }, 256, 256, &layer.Vertices, &layer.VertexCount, &layer.Indices, &layer.IndexCount);

        uint32_t fiber_count = (uint32_t)(2.0f*Tempest::MathPi*cur_radius/(2.0f*FiberRadius + FiberMinGap));
        if(fiber_count == 0)
        {
            Tempest::Matrix4 cylinder_mtx;
            cylinder_mtx.identity();
            cylinder_mtx.translate(yarn_pos);

            Tempest::RTSubmesh cylinder;
            cylinder.Material = &material;
            cylinder.BaseIndex = 0;
            cylinder.VertexCount = layer.IndexCount;
            cylinder.VertexOffset = 0;
			cylinder.Stride = stride;

            uint32_t stride = 6*sizeof(float);
            rt_scene->addTriangleMesh(cylinder_mtx, 1, &cylinder, layer.IndexCount, layer.Indices, layer.VertexCount*stride, layer.Vertices);

            continue;
        }

        for(uint32_t angle_idx = 0; angle_idx < fiber_count; ++angle_idx)
        {
            Tempest::Matrix4 cylinder_mtx;
            cylinder_mtx.identity();
            cylinder_mtx.translate(yarn_pos);
            cylinder_mtx.rotateZ(2.0f*Tempest::MathPi*angle_idx/fiber_count);

            Tempest::RTSubmesh cylinder;
            cylinder.Material = &material;
            cylinder.BaseIndex = 0;
            cylinder.VertexCount = layer.IndexCount;
            cylinder.VertexOffset = 0;
            cylinder.Stride = 6*sizeof(float);
            rt_scene->addTriangleMesh(cylinder_mtx, 1, &cylinder, layer.IndexCount, layer.Indices, layer.VertexCount*stride, layer.Vertices);
        }
    }
    /*/
    FiberLayer cyl_info;
    Tempest::Matrix4 place_bent_cyl;
    place_bent_cyl.identity();
    place_bent_cyl.translate(yarn_pos);
    //place_bent_cyl.translate(Tempest::Vector3{ YarnRadius * 2.0f, 0.0f, 0.0f });

    Tempest::TriangleTessellation(Tempest::Cylinder{ Tempest::Vector3{}, cyl_rad, FiberLength }, 256, 256, &cyl_info.Vertices, &cyl_info.VertexCount, &cyl_info.Indices, &cyl_info.IndexCount);

    Tempest::RTSubmesh cylinder;
    cylinder.Material = &material;
    cylinder.BaseIndex = 0;
    cylinder.VertexCount = cyl_info.IndexCount;
    cylinder.VertexOffset = 0;

    //rt_scene->addTriangleMesh(place_bent_cyl, 1, &cylinder, cyl_info.IndexCount, cyl_info.Indices, cyl_info.VertexCount*stride, cyl_info.Vertices, stride);

    rt_scene->addCylinder(Tempest::Cylinder{ yarn_pos, YarnRadius + FiberRadius, FiberLength }, &material);

    float blocker_size = std::max(FiberLength, 2.0f*YarnRadius + fabsf(FiberSkew));

   // rt_scene->addBlocker(target, Tempest::Vector3{0.0f, 0.0f, 1.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f}, Tempest::Vector2{blocker_size, blocker_size});
    //*/

	rt_scene->commitScene();

    Tempest::TimeQuery query;
    auto start = query.time();

    rt_scene->initWorkers();

    size_t tex_size = SamplesAzimuth*SamplesLong*image_width*image_height;
    auto final_image = std::unique_ptr<uint32_t[]>(new uint32_t[tex_size]);

    std::fill(final_image.get(), final_image.get() + tex_size, 0x00FF00FF);

    uint32_t image_size = image_width*image_height;
    
    for(uint32_t samp = 0, samp_end = SamplesLong*SamplesAzimuth; samp < samp_end; ++samp)
    {
        uint32_t next_samp = samp + 1;
        uint32_t next_samp_azimuth = next_samp % SamplesAzimuth;
        uint32_t next_samp_long = next_samp / SamplesAzimuth;
        uint32_t samp_azimuth = samp % SamplesAzimuth;
        uint32_t samp_long = samp / SamplesAzimuth;

    //    Tempest::Log(Tempest::LogLevel::Debug, rot_angle_azimuth, "(", samp_azimuth, "), ", rot_angle_long, "(", samp_long, ")");
        rot_angle_azimuth = Tempest::MathPi*(2.0f*next_samp_azimuth/(SamplesAzimuth - 1) - 1.0f);
        rot_angle_long = 0.5f*Tempest::MathPi*(2.0f*next_samp_long/(SamplesLong - 1) - 1.0f);
        rot.identity();
        rot.rotateX(rot_angle_long);
        rot.rotateZ(rot_angle_azimuth);

        Tempest::Matrix4 cur_view_proj = proj * view * rot;
        Tempest::Matrix4 cur_view_proj_inv = cur_view_proj.inverse();

        auto* frame_data = rt_scene->draw(image_width, image_height, cur_view_proj_inv);
        TGE_CHECK(frame_data->Backbuffer, "Invalid backbuffer");

        for(uint32_t row = 0; row < image_height; ++row)
        {
            std::copy_n(reinterpret_cast<uint32_t*>(frame_data->Backbuffer->getData()) + image_width * row, image_width, final_image.get() + image_width*((samp_long*image_height + row)*SamplesAzimuth + samp_azimuth));
        }
    }

    auto end = query.time();

    Tempest::Log(Tempest::LogLevel::Info, "Render time (ray tracing): ", end - start, "us");
   
    Tempest::TextureDescription tex_desc;
    tex_desc.Width = SamplesAzimuth*image_width;
    tex_desc.Height = SamplesLong*image_height;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    Tempest::SaveImage(tex_desc, final_image.get(), Tempest::Path("yarn.tga"));

    Tempest::DisplayImage(tex_desc, final_image.get());
}