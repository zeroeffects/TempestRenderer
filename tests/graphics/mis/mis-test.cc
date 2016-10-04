#include "tempest/utils/testing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/compute/ray-tracing-cuda-system.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"

#if CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#endif

const uint32_t ImageWidth = 640;
const uint32_t ImageHeight = 480;

TGE_TEST("Testing Multiple Importance Sampling")
{
    Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(28.0f, (float)ImageWidth / ImageHeight, 0.1f, 1000.0f);

    Tempest::Vector3 target{0.0f, -2.0f, 2.5f},
                     origin{0.0f, 2.0f, 15.0f},
                     up{0.0f, 1.0f, 0.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAt(origin, target, up);

    view_proj *= view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    Tempest::Vector3 target_in_view = view*up;

    Tempest::Matrix4 view_inv;
    view_inv = view.inverse();
    
	RAY_TRACING_SYSTEM rt_sys(ImageWidth, ImageHeight, view_proj_inv);
	auto rt_scene = rt_sys.getRayTracer();

    Tempest::Matrix4 world = Tempest::Matrix4::identityMatrix();

    float verts[] = 
    {
         4.0f, -3.99615f,  4.0667f, 0.0f, 0.0f, 0.0f,
         4.0f, -3.82069f,  3.08221f, 0.0f, 0.0f, 0.0f,
        -4.0f, -3.82069f,  3.08221f, 0.0f, 0.0f, 0.0f,
        -4.0f, -3.82069f,  3.08221f, 0.0f, 0.0f, 0.0f,
        -4.0f, -3.99615f,  4.0667f, 0.0f, 0.0f, 0.0f,
         4.0f, -3.99615f,  4.0667f, 0.0f, 0.0f, 0.0f,

		 4.0f, -3.73096f,  2.70046f, 0.0f, 0.0f, 0.0f,
         4.0f, -3.43378f,  1.74564f, 0.0f, 0.0f, 0.0f,
        -4.0f, -3.43378f,  1.74564f, 0.0f, 0.0f, 0.0f,
        -4.0f, -3.43378f,  1.74564f, 0.0f, 0.0f, 0.0f,
        -4.0f, -3.73096f,  2.70046f, 0.0f, 0.0f, 0.0f,
         4.0f, -3.73096f,  2.70046f, 0.0f, 0.0f, 0.0f,

         4.0f, -3.28825f,  1.36972f, 0.0f, 0.0f, 0.0f,
         4.0f, -2.83856f,  0.476536f, 0.0f, 0.0f, 0.0f,
        -4.0f, -2.83856f,  0.476536f, 0.0f, 0.0f, 0.0f,
        -4.0f, -2.83856f,  0.476536f, 0.0f, 0.0f, 0.0f,
        -4.0f, -3.28825f,  1.36972f, 0.0f, 0.0f, 0.0f,
         4.0f, -3.28825f,  1.36972f, 0.0f, 0.0f, 0.0f,

         4.0f, -2.70651f,  0.25609f, 0.0f, 0.0f, 0.0f,
         4.0f, -2.08375f, -0.526323f, 0.0f, 0.0f, 0.0f,
        -4.0f, -2.08375f, -0.526323f, 0.0f, 0.0f, 0.0f,
        -4.0f, -2.08375f, -0.526323f, 0.0f, 0.0f, 0.0f,
        -4.0f, -2.70651f,  0.25609f, 0.0f, 0.0f, 0.0f,
         4.0f, -2.70651f,  0.25609f, 0.0f, 0.0f, 0.0f,
    };

    int32_t indices[] = { 0, 1, 2, 3, 4, 5 };

    Tempest::RTMicrofacetMaterial plane_mtls[4];
    Tempest::RTSubmesh submeshes[4];
    
	uint32_t off = 0;
	float v1, v2, v3;
    for(size_t i = 0; i < 4; ++i)
    {
        auto& plane_mtl = plane_mtls[i];
        plane_mtl.Diffuse = Tempest::RGBToSpectrum(Tempest::Vector3{0.07f, 0.09f, 0.13f});
        plane_mtl.Specular = Tempest::RGBToSpectrum(Tempest::Vector3{1.0f, 1.0f, 1.0f});
        plane_mtl.SpecularPower.x = 300.0f + 30.0f*powf(15.0f, (float)i);

        auto& submesh = submeshes[i];
        submesh.BaseIndex = 0;
        submesh.Material = &plane_mtl;
        submesh.VertexCount = 6;
        submesh.VertexOffset = off*sizeof(float);
		submesh.Stride = 6*sizeof(float);

		v1 = verts[off++];
		v2 = verts[off++];
		v3 = verts[off++];
		Tempest::Vector3 p1{v1, v2, v3};
		
		uint32_t norm_off = off;

		off += 3;

		v1 = verts[off++];
		v2 = verts[off++];
		v3 = verts[off++];
		Tempest::Vector3 p2{v1, v2, v3};
		
		off += 3;

		v1 = verts[off++];
		v2 = verts[off++];
		v3 = verts[off++];
		Tempest::Vector3 p3{v1, v2, v3};
		
		Tempest::Vector3 norm(Cross(p3 - p2, p1 - p2));
		NormalizeSelf(&norm);

		for(size_t j = 0; j < 6; ++j)
		{
			Tempest::CopyVec3ToFloatArray(norm, verts + norm_off);
			norm_off += 6;
		}

		off = norm_off - 3;
    }

    rt_scene->addTriangleMesh(world, 4, submeshes, submeshes[0].VertexCount, indices, sizeof(verts), verts);
	
    Tempest::SphereAreaLight* area_light1 = new Tempest::SphereAreaLight;
    area_light1->SphereShape.Center = Tempest::Vector3{-3.75f, 0.0f, 0.0f};
    area_light1->SphereShape.Radius = 0.03333f;
    area_light1->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{901.803f, 901.803f, 0.0f});

    rt_scene->addSphereLightSource(area_light1);
	//*
    Tempest::SphereAreaLight* area_light2 = new Tempest::SphereAreaLight;
    area_light2->SphereShape.Center = Tempest::Vector3{-1.25f, 0.0f, 0.0f};
    area_light2->SphereShape.Radius = 0.1f;
    area_light2->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{0.0f, 100.0f, 0.0f});

    rt_scene->addSphereLightSource(area_light2);
	
    Tempest::SphereAreaLight* area_light3 = new Tempest::SphereAreaLight;
    area_light3->SphereShape.Center = Tempest::Vector3{1.25f, 0.0f, 0.0f};
    area_light3->SphereShape.Radius = 0.3f;
    area_light3->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 11.1111f});

    rt_scene->addSphereLightSource(area_light3);
	
    Tempest::SphereAreaLight* area_light4 = new Tempest::SphereAreaLight;
    area_light4->SphereShape.Center = Tempest::Vector3{3.75f, 0.0f, 0.0f};
    area_light4->SphereShape.Radius = 0.9f;
    area_light4->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{1.23457f, 0.0f, 0.0f});

    rt_scene->addSphereLightSource(area_light4);

    //*/
    rt_scene->setSamplesGlobalIllumination(4);
    rt_scene->setSamplesLocalAreaLight(4);
    rt_scene->setMaxRayDepth(1);
    rt_scene->setRussianRoulette(0.9f);

    //rt_scene->setFillColor(Tempest::Vector3(0.1f, 0.1f, 0.1f));

    rt_scene->commitScene();

    Tempest::TimeQuery query;
    auto start = query.time();

    rt_sys.startRendering();

	rt_sys.completeFrame();

    auto end = query.time();

    Tempest::Log(Tempest::LogLevel::Info, "Render time (ray tracing): ", end - start, "us");

	rt_sys.saveImage(Tempest::Path("test.tga"));
    rt_sys.displayImage();
}