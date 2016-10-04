#include "tempest/utils/testing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/math/shapes.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/patterns.hh"

const float FiberRadius = 0.05f;
const float FiberLength = 5.0f;
const float AngularSpeed = 0.1f;
const float YarnRadius = 1.0f;
const float FiberMinGap = 0.0f;
const float FiberSkew = -5.0f;

TGE_TEST("Testing tessellating shapes")
{
    float cyl_rad = YarnRadius + FiberRadius;
	float proj_horiz_span = 2.0f*cyl_rad;
	float proj_vert_span = FiberLength;

    float scale_image = 100;
    uint32_t image_width = uint32_t(proj_horiz_span*scale_image);
	uint32_t image_height = uint32_t(proj_vert_span*scale_image);

    Tempest::Matrix4 view_proj = Tempest::OrthoMatrix(-proj_horiz_span, proj_horiz_span, -proj_vert_span, proj_vert_span, 0.1f, 1000.0f);

    Tempest::Vector3 target{0.0f, 15.0f, 0.0f},
                     origin{0.0f, 0.0f, 0.0f},
                     up{0.0f, 0.0f, 1.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAt(origin, target, up);

    view_proj *= view;

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
	dir_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{1.0f, 1.0f, 1.0f});
	rt_scene->addLightSource(dir_light);

	rt_scene->setSamplesCamera(64);
    rt_scene->setSamplesLocalAreaLight(1);
    rt_scene->setSamplesGlobalIllumination(1);
    rt_scene->setMaxRayDepth(6);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::RGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    Tempest::RTMicrofacetMaterial material;
	memset(&material, 0x0, sizeof(material));
	
    Tempest::Spectrum color = Tempest::RGBToSpectrum(Tempest::Vector3{ 0.1f, 1.0f, 0.1f });

    material.Model = Tempest::IlluminationModel::BlinnPhong;
	material.Diffuse = color*0.0f;
    material.Specular = color*1.0f;
    material.SpecularPower = Tempest::Vector2{ 100.0f, 100.0f };
	material.Fresnel.x = 0.5f;
    material.setup();

    struct FiberLayer
    {
        Tempest::ShapeVertex* Vertices = nullptr;
        uint32_t       VertexCount = 0;
        int32_t*       Indices = nullptr;
        uint32_t       IndexCount = 0;

        ~FiberLayer()
        {
            delete[] Vertices;
            delete[] Indices;
        }
    };

    const uint32_t stride = 6*sizeof(float);
    //*
    uint32_t layer_count = (uint32_t)(YarnRadius/(2.0f*FiberRadius + FiberMinGap));
    std::unique_ptr<FiberLayer[]> layers(new FiberLayer[layer_count]);

    Tempest::Vector3 yarn_pos = target;
    yarn_pos.x -= cyl_rad;

    for(uint32_t layer_idx = 0; layer_idx < layer_count; ++layer_idx)
    {
        auto& layer = layers[layer_idx];

        const float cur_radius = layer_count > 1 ? ((float)layer_idx/(layer_count - 1))*YarnRadius : YarnRadius;

        Tempest::TriangleTessellation(Tempest::HelixCylinder{ Tempest::Cylinder{ Tempest::Vector3{}, FiberRadius, FiberLength }, AngularSpeed, cur_radius }, 256, 256, &layer.Vertices, &layer.VertexCount, &layer.Indices, &layer.IndexCount);
        
        const uint32_t fiber_count = (uint32_t)(2.0f*Tempest::MathPi*cur_radius/(2.0f*FiberRadius + FiberMinGap));
        const uint32_t slice_size = layer.VertexCount*stride;

        if(fiber_count == 0)
        {
            Tempest::RTSubmesh cylinder;
            cylinder.Material = &material;
            cylinder.BaseIndex = 0;
            cylinder.VertexCount = layer.IndexCount;
            cylinder.VertexOffset = 0;
			cylinder.Stride = stride;

            auto* vert = layer.Vertices;

            Tempest::CurveSkewYZ(layer.Vertices, layer.VertexCount, FiberSkew, FiberLength, layer.Vertices);

            Tempest::Matrix4 place_mtx;
            place_mtx.identity();
            place_mtx.translate(yarn_pos);

            rt_scene->addTriangleMesh(place_mtx, 1, &cylinder, layer.IndexCount, layer.Indices, slice_size, layer.Vertices);

            continue;
        }

        std::unique_ptr<Tempest::ShapeVertex[]> old_verts(layer.Vertices);
        layer.Vertices = new Tempest::ShapeVertex[fiber_count*slice_size];
        for(uint32_t angle_idx = 0; angle_idx < fiber_count; ++angle_idx)
        {
            Tempest::Matrix4 fiber_mtx;
            fiber_mtx.identity();
            fiber_mtx.rotateZ(2.0f*Tempest::MathPi*angle_idx/fiber_count);

            Tempest::RTSubmesh cylinder;
            cylinder.Material = &material;
            cylinder.BaseIndex = 0;
            cylinder.VertexCount = layer.IndexCount;
            cylinder.VertexOffset = 0;
			cylinder.Stride = stride;

            auto* vert = layer.Vertices + angle_idx*slice_size;
            
            for(uint32_t i = 0; i < layer.VertexCount; ++i)
            {
                vert[i].Position = fiber_mtx.transformRotate(old_verts[i].Position);
                vert[i].Normal = fiber_mtx.transformRotate(old_verts[i].Normal);
            }
            
            Tempest::CurveSkewYZ(vert, layer.VertexCount, FiberSkew, FiberLength, vert);

            Tempest::Matrix4 place_mtx;
            place_mtx.identity();
            place_mtx.translate(yarn_pos);

            rt_scene->addTriangleMesh(place_mtx, 1, &cylinder, layer.IndexCount, layer.Indices, slice_size, layer.Vertices + slice_size*angle_idx);
        }
    }
    //*/
    FiberLayer cyl_info;
    {
    Tempest::Matrix4 place_bent_cyl;
    place_bent_cyl.identity();
    place_bent_cyl.translate(Tempest::Vector3{ yarn_pos.x + 2.0f*cyl_rad, yarn_pos.y, yarn_pos.z });
    //place_bent_cyl.translate(Tempest::Vector3{ YarnRadius * 2.0f, 0.0f, 0.0f });

    Tempest::TriangleTessellation(Tempest::Cylinder{ Tempest::Vector3{}, cyl_rad, FiberLength }, 256, 256, &cyl_info.Vertices, &cyl_info.VertexCount, &cyl_info.Indices, &cyl_info.IndexCount);
    Tempest::CurveSkewYZ(cyl_info.Vertices, cyl_info.VertexCount, FiberSkew, FiberLength, cyl_info.Vertices);

    Tempest::RTSubmesh cylinder;
    cylinder.Material = &material;
    cylinder.BaseIndex = 0;
    cylinder.VertexCount = cyl_info.IndexCount;
    cylinder.VertexOffset = 0;
	cylinder.Stride = stride;

    rt_scene->addTriangleMesh(place_bent_cyl, 1, &cylinder, cyl_info.IndexCount, cyl_info.Indices, cyl_info.VertexCount*stride, cyl_info.Vertices);
    }

    /*
	Tempest::Vector3 cyl_coord = target;
	cyl_coord.z -= 6.0f;

	rt_scene->addCylinder(Tempest::Cylinder{ cyl_coord, YarnRadius + FiberRadius, 1.0f }, &material);
	
	cyl_coord.z = target.z + 6.0f;
	rt_scene->addCylinder(Tempest::Cylinder{ cyl_coord, YarnRadius + FiberRadius, 1.0f }, &material);
    */

    float blocker_size = std::max(FiberLength, 2.0f*YarnRadius + fabsf(FiberSkew));

    rt_scene->addBlocker(target, Tempest::Vector3{0.0f, 0.0f, 1.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f}, Tempest::Vector2{blocker_size, blocker_size});

	rt_scene->commitScene();

    Tempest::TimeQuery query;
    auto start = query.time();

    rt_scene->initWorkers();

    auto* frame_data = rt_scene->drawOnce();
    TGE_CHECK(frame_data->Backbuffer, "Invalid backbuffer");

    auto end = query.time();

    Tempest::Log(Tempest::LogLevel::Info, "Render time (ray tracing): ", end - start, "us");
   
    auto* backbuffer = frame_data->Backbuffer.get();
    Tempest::SaveImage(backbuffer->getHeader(), backbuffer->getData(), Tempest::Path("yarn.tga"));
    Tempest::DisplayImage(backbuffer->getHeader(), backbuffer->getData());
}