#include "tempest/utils/testing.hh"
#include "tempest/mesh/sslbvh2.hh"
#include "tempest/mesh/lbvh2.hh"
#include "tempest/math/triangle.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/mesh/obj-loader.hh"

#include <memory>
#include <algorithm>

#if CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#   define RASTERIZER_FUNCTION __device__
#   define RASTERIZER Tempest::RasterizerCuda
const uint32_t ImageWidth = 800;
const uint32_t ImageHeight = 800;
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#   define RASTERIZER_FUNCTION
#   define RASTERIZER Tempest::Rasterizer
const uint32_t ImageWidth = 400;
const uint32_t ImageHeight = 400;
#endif

const uint32_t PointCount = 1024;
const uint32_t RandomSamples = 1024;
TGE_TEST("Testing BVH construction and intersection functions")
{
    uint32_t seed = 1;
    std::unique_ptr<Tempest::Vector2[]> points(new Tempest::Vector2[PointCount]);
    std::generate(points.get(), points.get() + PointCount, [&seed](){ return Tempest::Vector2{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) }; });

    uint32_t index_count;
    auto index_list = CREATE_SCOPED(uint32_t*, ::free);
    Tempest::DelaunayTriangulation(points.get(), PointCount, &index_list, &index_count);

    std::unique_ptr<Tempest::LBVH2Node<Tempest::AABB2>> interm_nodes(Tempest::GenerateTriangleNodes<Tempest::AABB2>(points.get(), PointCount, index_list, index_count));
    std::unique_ptr<Tempest::LBVH2Node<Tempest::AABB2>> lbvh(Tempest::GenerateLBVH(interm_nodes.get(), PointCount));
    std::unique_ptr<Tempest::SimpleStacklessLBVH2Node<Tempest::AABB2>> stackless_lbvh(Tempest::GenerateSSLBVH(interm_nodes.get(), PointCount));

    Tempest::IntersectTriangleQuery2D intersect_tri = { points.get(), index_list };

    for(uint32_t idx = 0; idx < RandomSamples; ++idx)
    {
        Tempest::Vector2 point{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) };

        auto intersect0 = Tempest::IntersectLBVHNodeSingle(lbvh.get(), 0, point, intersect_tri);
        auto prim_id0 = intersect_tri.PrimitiveID;
        auto intersect1 = Tempest::IntersectSSLBVHNodeSingle(stackless_lbvh.get(), point, intersect_tri);
        auto prim_id1 = intersect_tri.PrimitiveID;

        TGE_CHECK((intersect0 == 0 && intersect1 == 0) || prim_id0 == prim_id1, "Invalid intersection and LBVH implementation");
    }

	// Test whether it produces the same results as the ray tracer
	Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(45.0f, (float)ImageWidth / ImageHeight, 0.1f, 1000.0f);
    
    Tempest::Vector3 target{0.0f, 50.0f, 0.0f},
                     origin{0.0f, 170.0f, 100.0f},
                     up{0.0f, 1.0f, 0.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAtUpTarget(origin, target, up);

    view_proj *= view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

	RAY_TRACING_SYSTEM rt_sys(ImageWidth, ImageHeight, view_proj_inv);

	auto* rt_scene = rt_sys.getRayTracer();

	Tempest::RTMeshBlob mesh_blob;
    auto status = Tempest::LoadObjFileStaticRTGeometry(TEST_ASSETS_DIR "/teapot/teapot.obj", nullptr, &mesh_blob);
    TGE_CHECK(status, "Failed to load test assets");

	Tempest::RTMicrofacetMaterial emissive_material;
	emissive_material.Model = Tempest::IlluminationModel::Emissive;
	emissive_material.Diffuse = { 1.0f, 1.0f, 1.0f };
    emissive_material.Specular = {};
    emissive_material.setup();

	uint32_t image_area = ImageWidth*ImageHeight;

	std::unique_ptr<uint32_t[]> manual_trace_image(new uint32_t[image_area]);
	std::fill(manual_trace_image.get(), manual_trace_image.get() + image_area, 0xFF000000U);

	for(uint32_t submesh_idx = 0, submesh_count = mesh_blob.SubmeshCount;
		submesh_idx < submesh_count; ++submesh_idx)
	{
		auto& submesh = mesh_blob.Submeshes[submesh_idx];
		submesh.Material = &emissive_material;

		size_t max_node_count = (2*submesh.VertexCount - 1);
		size_t bvh_size = max_node_count*sizeof(Tempest::SimpleStacklessLBVH2Node<Tempest::AABBUnaligned>);

		auto tri_count = submesh.VertexCount/3;

		uint8_t* submesh_vert_ptr = reinterpret_cast<uint8_t*>(&mesh_blob.VertexData.front()) + submesh.VertexOffset;
		auto submesh_ind_ptr = reinterpret_cast<uint32_t*>(&mesh_blob.IndexData.front() + submesh.BaseIndex);
		uint32_t submesh_vert_count = (static_cast<uint32_t>(mesh_blob.VertexData.size()) - submesh.VertexOffset)/submesh.Stride;
		std::unique_ptr<Tempest::LBVH2Node<Tempest::AABBUnaligned>> nodes(Tempest::GenerateTriangleNodes<Tempest::AABBUnaligned>(submesh_vert_ptr, submesh_vert_count, submesh_ind_ptr, tri_count, submesh.Stride));
	
		std::unique_ptr<Tempest::SimpleStacklessLBVH2Node<Tempest::AABBUnaligned>> bvh(GenerateSSLBVH(nodes.get(), tri_count));

		for(uint32_t y = 0; y < ImageHeight; ++y)
			for(uint32_t x = 0; x < ImageWidth; ++x)
			{
				Tempest::IntersectTriangleQuery3DCull intersect_tri{ submesh_vert_ptr, submesh.Stride, submesh_ind_ptr };

				Tempest::Vector4 screen_tc = Tempest::Vector4{2.0f*x/(ImageWidth - 1) - 1.0f, 2.0f*y/(ImageHeight - 1) - 1.0f, -1.0f, 1.0};

				Tempest::Vector4 pos_start = view_proj_inv*screen_tc;

				screen_tc.z = 1.0f;
				Tempest::Vector4 pos_end = view_proj_inv*screen_tc;

				auto start_ray_pos = Tempest::ToVector3(pos_start);
				auto end_ray_pos = Tempest::ToVector3(pos_end);

				auto inc_light = Normalize(end_ray_pos - start_ray_pos);

				Tempest::RayIntersectData intersect_data{ inc_light, start_ray_pos, 0.0f, INFINITY};

				if(Tempest::IntersectSSLBVHNodeSingle(bvh.get(), intersect_data, intersect_tri))
					manual_trace_image[y*ImageWidth + x] |= ~0U;
			}
		Tempest::AABBUnaligned aabb;

		aabb.MinCorner = Vector3Min(bvh->Bounds.MinCorner, aabb.MinCorner);
		aabb.MaxCorner = Vector3Max(bvh->Bounds.MaxCorner, aabb.MaxCorner);
	}

	Tempest::TextureDescription manual_tex_desc;
	manual_tex_desc.Width = ImageWidth;
	manual_tex_desc.Height = ImageHeight;
	manual_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;
	Tempest::SaveImage(manual_tex_desc, manual_trace_image.get(), Tempest::Path("manual-teapot-render.tga"));

	rt_scene->addTriangleMesh(Tempest::Matrix4::identityMatrix(), mesh_blob.SubmeshCount, mesh_blob.Submeshes,
							  mesh_blob.IndexData.size()/3, &mesh_blob.IndexData.front(), mesh_blob.VertexData.size(), &mesh_blob.VertexData.front());

	rt_scene->commitScene();

	rt_sys.startRendering();

	rt_sys.completeFrame();
	
	rt_sys.saveImage(Tempest::Path("teapot-render.tga"));

	auto ray_traced_tex = rt_sys.getLastFrameTexture();
	auto ray_traced_image = ray_traced_tex->getData();

	uint32_t different_pixels = 0;
	for(uint32_t pixel_idx = 0; pixel_idx < image_area; ++pixel_idx)
	{
		if(reinterpret_cast<const uint32_t*>(ray_traced_image)[pixel_idx] != manual_trace_image[pixel_idx])
			++different_pixels;
	}

	TGE_CHECK(different_pixels < 10, "Too different results - probably broken intersection test");
}