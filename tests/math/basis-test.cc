#include "tempest/utils/testing.hh"
#include "tempest/math/numerical-methods.hh"
#include "tempest/math/simple-basis.hh"
#include "tempest/math/spherical-harmonics.hh"
#include "tempest/math/shapes.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/math/shapes.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"

const uint32_t SampleCount = 1024;

TGE_TEST("Testing some simple basis functions")
{
    auto area = Tempest::StratifiedMonteCarloIntegratorHemisphere(SampleCount,
                    [](const Tempest::Vector3& dir)
                    {
                        return 1.0f/(2.0f*Tempest::MathPi);
                    });

    TGE_CHECK(Tempest::ApproxEqual(area, 1.0f, 1e-3f), "Broken integrator");

    auto coefs = Tempest::StratifiedMonteCarloIntegratorHemisphere<Tempest::Vector3>(SampleCount,
                    [](const Tempest::Vector3& dir)
                    {
                        return Tempest::ThreeBasis(dir);
                    });

    auto area_proj = Tempest::StratifiedMonteCarloIntegratorHemisphere(SampleCount,
                    [coefs](const Tempest::Vector3& dir)
                    {
                        return Tempest::Dot(Tempest::ThreeBasis(dir), coefs);
                    });

    TGE_CHECK(Tempest::ApproxEqual(area_proj, 1.0f, 1e-3f), "Broken integrator");

    float scale_image = 100;
    int32_t image_width = 150;
	int32_t image_height = 400;
    uint32_t tesselate = 128;

    auto max_dim = std::max(image_width, image_height);

    //Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(40.0f, (float)image_width/image_height, 0.1f, 1000.0f);
    Tempest::Matrix4 view_proj = Tempest::OrthoMatrix(-(float)2.0f*image_width/max_dim, (float)1.5f*image_width/max_dim, -(float)1.5f*image_height/max_dim, (float)1.5f*image_height/max_dim, 0.1f, 1000.0f);

    Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                     origin{2.0f, 2.0f, 0.5f},
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

    Tempest::Sphere sphere{ { 0.0f, 0.0f, 0.0f }, 1.0f };

    /*
    Tempest::SphereAreaLight* area_light = new Tempest::SphereAreaLight;
    area_light->SphereShape.Center = Tempest::Vector3{0.0f, 0.0f, 0.0f};
    area_light->SphereShape.Radius = 0.1f;
    area_light->Radiance = Tempest::Vector3{5000.0f, 5000.0f, 5000.0f};

    rt_scene->addLightSource(area_light);
	*/

	Tempest::DirectionalLight* dir_light = new Tempest::DirectionalLight;
	dir_light->Direction = Tempest::Normalize(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });
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
	
    Tempest::Spectrum color = Tempest::RGBToSpectrum(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });

    material.Model = Tempest::IlluminationModel::BlinnPhong;
	material.Diffuse = color*0.5f;
    material.Specular = color*1.0f;
    material.SpecularPower = Tempest::Vector2{ 100.0f, 100.0f };
	material.Fresnel.x = 0.5f;
    material.setup();

    Tempest::ScopedObject<Tempest::ShapeVertex*, Tempest::DefaultArrayDeleter<Tempest::ShapeVertex>> verts;
    Tempest::ScopedObject<int32_t*, Tempest::DefaultArrayDeleter<int32_t>> indices;

    uint32_t vert_count, index_count;

    for(int32_t cur_degree = 0; cur_degree < 3; ++cur_degree)
    {
        Tempest::TriangleTessellation(sphere, 2*tesselate, tesselate, &verts, &vert_count, &indices, &index_count,
                                        [cur_degree](const Tempest::Vector3& dir)
                                        {
                                            auto basis = Tempest::ThreeBasis(dir);
                                            return Array(basis)[cur_degree];
                                        });

        Tempest::RTSubmesh submesh;
        submesh.BaseIndex = 0;
        submesh.Material = &material;
        submesh.Stride = sizeof(Tempest::ShapeVertex);
        submesh.VertexCount = index_count;
        submesh.VertexOffset = 0;

        //rt_scene->addSphere(sphere, &material);
    
        Tempest::MeshOptions mesh_opts;
        mesh_opts.TwoSided = false;

        Tempest::Matrix4 mat;
        mat.identity();
        mat.translate(Tempest::Vector3{ (float)0.0f, 0.0f, cur_degree - 1.0f});

        rt_scene->addTriangleMesh(mat, 1, &submesh, index_count/3, indices, vert_count*sizeof(Tempest::ShapeVertex), verts, &mesh_opts);
    }

    rt_scene->setRenderMode(Tempest::RenderMode::DebugNormals);

	rt_scene->commitScene();

    Tempest::TimeQuery query;
    auto start = query.time();

    rt_scene->initWorkers();

    auto* frame_data = rt_scene->drawOnce();
    TGE_CHECK(frame_data->Backbuffer, "Invalid backbuffer");

    auto end = query.time();

    Tempest::Log(Tempest::LogLevel::Info, "Render time (ray tracing): ", end - start, "us");
   
    auto* backbuffer = frame_data->Backbuffer.get();
    Tempest::SaveImage(backbuffer->getHeader(), backbuffer->getData(), Tempest::Path("three-basis.png"));
    Tempest::DisplayImage(backbuffer->getHeader(), backbuffer->getData());
}