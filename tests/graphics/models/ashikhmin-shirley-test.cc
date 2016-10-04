#include "tempest/utils/testing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/refractive-indices.hh"

const float SphereGap = 0.1f;
const float SphereRadius = 1.0f;
const size_t SphereCount = 9;

TGE_TEST("Testing Ashikhmin-Shirley anisotropic model")
{
    uint32_t image_width = 500;
    uint32_t image_height = 500;

    float span = SphereCount*0.5f*2.0f*(SphereRadius + SphereGap);

    //Tempest::Matrix4 view_proj = Tempest::OrthoMatrix(-span, span, -span, span, 0.1f, 1000.0f);
    Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(90.0f, (float)image_width / image_height, 0.1f, 1000.0f);

    Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                     origin{0.0f, 10.0f, 0.0f},
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

    Tempest::DirectionalLight* dir_light = new Tempest::DirectionalLight;
    dir_light->Direction = Tempest::Vector3{1.0f, 1.0f, 0.0f};
    NormalizeSelf(&dir_light->Direction);
    dir_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{2.0f, 2.0f, 2.0f});

    float verts[] = 
    {
         100.0f, -0.11f, -100.0f, 0.0f, 8.0f, 0.0f, 1.0f, 0.0f,
         100.0f, -0.11f,  100.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        -100.0f, -0.11f,  100.0f, 8.0f, 8.0f, 0.0f, 1.0f, 0.0f,
        -100.0f, -0.11f, -100.0f, 8.0f, 0.0f, 0.0f, 1.0f, 0.0f
    };

    int32_t indices[] = { 0, 3, 1, 3, 2, 1 };

    Tempest::Matrix4 plane_mat;
    plane_mat.identity();
    plane_mat.translateY(0.2f);

    std::unique_ptr<Tempest::Texture> diffuse_map(Tempest::LoadImage(Tempest::Path(TEST_ASSETS_DIR "/scarf/data/concrete.png")));

    Tempest::RTMicrofacetMaterial plane_material;
    plane_material.DiffuseMap = rt_scene->bindTexture(diffuse_map.get());
    plane_material.Diffuse = Tempest::RGBToSpectrum(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });
    plane_material.Specular = Tempest::RGBToSpectrum(Tempest::Vector3{ 0.0f, 0.0f, 0.0f });
    plane_material.SpecularPower.x = 1.0f;
    plane_material.setup();

    Tempest::RTSubmesh plane_submesh;
    plane_submesh.Material = &plane_material;
    plane_submesh.BaseIndex = 0;
    plane_submesh.VertexCount = 6;
    plane_submesh.VertexOffset = 0;
	plane_submesh.Stride = 8*sizeof(float);

    rt_scene->addTriangleMesh(plane_mat, 1, &plane_submesh, plane_submesh.VertexCount, indices, sizeof(verts), verts);

    rt_scene->addLightSource(dir_light);
    rt_scene->setSamplesGlobalIllumination(64);
    rt_scene->setMaxRayDepth(1);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::RGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

   
    Tempest::RTMicrofacetMaterial sphere_material;
    sphere_material.Model = Tempest::IlluminationModel::AshikhminShirley;
    sphere_material.Diffuse = Tempest::RGBToSpectrum(Tempest::Vector3{ 0.0f, 0.0f, 0.0f });
    sphere_material.Specular = Tempest::RGBToSpectrum(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });
    sphere_material.SpecularPower.x = 100.0f;
    sphere_material.SpecularPower.y = 100.0f;
	sphere_material.Fresnel = Tempest::IronRefractiveIndex;
    sphere_material.setup();

    for(size_t i = 0; i < SphereCount; ++i)
    {
        Tempest::Vector3 sphere_pos{ target.x + (i - (SphereCount - 1)*0.5f)*(2.0f*SphereRadius + SphereGap), target.y + SphereRadius, target.z };
    
        Tempest::Sphere sphere_geom { sphere_pos, SphereRadius };
        rt_scene->addSphere(sphere_geom, &sphere_material);
    }

    rt_scene->commitScene();

    Tempest::TimeQuery query;
    auto start = query.time();

    rt_scene->initWorkers();
	
    auto* frame_data = rt_scene->drawOnce();
    TGE_CHECK(frame_data->Backbuffer, "Invalid backbuffer");

    auto end = query.time();

    Tempest::Log(Tempest::LogLevel::Info, "Render time (ray tracing): ", end - start, "us");

    auto* backbuffer = frame_data->Backbuffer.get();
    Tempest::SaveImage(backbuffer->getHeader(), backbuffer->getData(), Tempest::Path("test.tga"));
    Tempest::DisplayImage(backbuffer->getHeader(), backbuffer->getData());
}