#include "tempest/utils/testing.hh"
#include "tempest/mesh/obj-loader.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"
#include "tempest/math/matrix3.hh"

const uint32_t RandomSamples = 16;

TGE_TEST("Testing how consistent are the consistent normals")
{
	Tempest::RTMeshBlob mesh_blob;
	uint32_t flags = Tempest::TEMPEST_OBJ_LOADER_GENERATE_CONSISTENT_NORMALS;
    auto status = Tempest::LoadObjFileStaticRTGeometry(TEST_ASSETS_DIR "/cloth/clothhd.obj", nullptr, &mesh_blob, flags);
    TGE_CHECK(status, "Failed to load test assets");

	unsigned seed = 1;

	for(uint32_t submesh_idx = 0; submesh_idx < mesh_blob.SubmeshCount; ++submesh_idx)
	{
		auto& submesh = mesh_blob.Submeshes[submesh_idx];

		if(submesh.Stride == sizeof(Tempest::PcNFormat))
			continue;

		TGE_CHECK(sizeof(Tempest::PTcNFormat) == submesh.Stride, "invalid mesh stride");

		for(uint32_t idx = submesh.BaseIndex, idx_end = submesh.BaseIndex + submesh.VertexCount; idx < idx_end;)
		{
			auto i0 = mesh_blob.IndexData[idx++];
			auto i1 = mesh_blob.IndexData[idx++];
			auto i2 = mesh_blob.IndexData[idx++];

			auto& v0 = reinterpret_cast<Tempest::PTcNFormat&>(mesh_blob.VertexData[submesh.VertexOffset + i0*submesh.Stride]);
			auto& v1 = reinterpret_cast<Tempest::PTcNFormat&>(mesh_blob.VertexData[submesh.VertexOffset + i1*submesh.Stride]);
			auto& v2 = reinterpret_cast<Tempest::PTcNFormat&>(mesh_blob.VertexData[submesh.VertexOffset + i2*submesh.Stride]);

			auto edge0 = v0.Position - v1.Position,
				 edge1 = v2.Position - v1.Position;

			auto geom_norm = Normalize(Tempest::Cross(edge1, edge0));

			float cos_norm0 = Tempest::Dot(geom_norm, v0.Normal),
				  cos_norm1 = Tempest::Dot(geom_norm, v1.Normal),
				  cos_norm2 = Tempest::Dot(geom_norm, v2.Normal);

			TGE_CHECK(cos_norm0 > 0.0f && cos_norm1 > 0.0f && cos_norm2 > 0.0f, "invalid normal");

			for(uint32_t sample_idx = 0; sample_idx < RandomSamples; ++sample_idx)
			{
				Tempest::Vector3 orig_dir;
				do
				{
					orig_dir = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
				} while(orig_dir.z < 0.01f);

				Tempest::Matrix3 basis;
				basis.makeBasis(geom_norm);

				auto dir = basis.transform(orig_dir);

				TGE_CHECK(Tempest::Dot(dir, basis.normal()) > 0.0f, "Bad basis");

				auto cons_norm0 = Tempest::ComputeConsistentNormal(dir, v0.Normal, v0.InterpolationConstant);
				TGE_CHECK(Tempest::Dot(cons_norm0, dir) > 0.0f, "Inconsistent normal");

				auto cons_norm1 = Tempest::ComputeConsistentNormal(dir, v1.Normal, v1.InterpolationConstant);
				TGE_CHECK(Tempest::Dot(cons_norm1, dir) > 0.0f, "Inconsistent normal");

				auto cons_norm2 = Tempest::ComputeConsistentNormal(dir, v2.Normal, v2.InterpolationConstant);
				TGE_CHECK(Tempest::Dot(cons_norm2, dir) > 0.0f, "Inconsistent normal");
			}
		}
	}

}