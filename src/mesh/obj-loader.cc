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

#include <vector>
#include <algorithm>

#include "tempest/mesh/obj-loader.hh"
#include "tempest/math/vector4.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/graphics/opengl-backend/gl-command-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-shader.hh"
#include "tempest/graphics/opengl-backend/gl-backend.hh"
#include "tempest/utils/interleave-vertices.hh"
#include "tempest/mesh/obj-loader-driver.hh"
#include "tempest/mesh/obj-mtl-loader.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/state-object.hh"
#include "tempest/texture/texture-table.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"
#include "tempest/math/triangle.hh"

namespace Tempest
{
class FileLoader;

template<class TBackend> class TextureTable;

RTMeshBlob::~RTMeshBlob()
{
	for(auto& tex : TextureData)
	{
		delete tex;
	}
	delete[] Submeshes;
	delete[] Materials;
}

static void InterleaveInterm(ObjLoader::Driver& obj_loader_driver, const ObjLoader::GroupHeader& hdr, size_t pos_size, size_t tc_size, size_t norm_size,
                             std::vector<uint16_t>* res_inds, std::vector<char>* res_data, uint32_t* stride, std::vector<int32_t>* res_inds32, uint32_t flags = 0)
{
    int32_t        strides[5] = {};
    const int32_t* inds[5] = {};
    const char*  verts[5] = {};

	std::vector<Vector3> gen_tans;
    std::vector<Vector3> gen_norms;
	std::vector<float> gen_coefs;
	std::vector<int32_t> gen_coefs_inds;
    std::vector<int32_t> gen_inds;

    *stride = 0;

    size_t num = 0, ind_count = 0;
    auto& pos = obj_loader_driver.getPositions();
    auto& tc = obj_loader_driver.getTexCoords();
    auto& norm = obj_loader_driver.getNormals();
    
    auto& pos_ind = obj_loader_driver.getPositionIndices();
    auto& tc_ind = obj_loader_driver.getTexCoordIndices();
    auto& norm_ind = obj_loader_driver.getNormalIndices();
    TGE_ASSERT(pos_size, "Position indices be greater than zero");
    if(pos_size == 0)
        return;

    verts[num] = reinterpret_cast<const char*>(&pos.front());
    inds[num] = &pos_ind[hdr.PositionStart];
    *stride += strides[num++] = sizeof(pos.front());
    ind_count = pos_size;
    
	int32_t min_ind = std::numeric_limits<int32_t>::max(), max_ind = 0;
	if(((flags & TEMPEST_OBJ_LOADER_GENERATE_TANGENTS) && tc_size != 0) ||
		norm_size == 0)
	{
		gen_inds.resize(pos_size);
	
		for(size_t i = hdr.PositionStart, iend = hdr.PositionStart + pos_size; i < iend; ++i)
		{
			auto ind = pos_ind[i];
			if(min_ind > ind)
				min_ind = ind;
			if(max_ind < ind)
				max_ind = ind;
		}

		for(size_t pos_idx = hdr.PositionStart, tc_idx = hdr.TexCoordStart, gen_idx = 0; gen_idx < pos_size;)
		{
			auto pos_prev_idx = pos_ind[pos_idx++];
			gen_inds[gen_idx++] = pos_prev_idx - min_ind;

			auto pos_current_idx = pos_ind[pos_idx++];
			gen_inds[gen_idx++] = pos_current_idx - min_ind;

			auto pos_next_idx = pos_ind[pos_idx++];
			gen_inds[gen_idx++] = pos_next_idx - min_ind;
		}
	}
	
    if(tc_size != 0)
    {
        verts[num] = reinterpret_cast<const char*>(&tc.front());
        inds[num] = &tc_ind[hdr.TexCoordStart];
        *stride += strides[num++] = sizeof(tc.front());
        TGE_ASSERT(ind_count == 0 || ind_count == tc_size, "Indices should be the same"); ind_count = tc_size;

		if(flags & TEMPEST_OBJ_LOADER_GENERATE_TANGENTS)
		{
			gen_tans.resize(max_ind - min_ind + 1, Vector3{0.0f, 0.0f, 0.0f});

//#define TANGENT_POWER_ITERATION

        #ifdef TANGENT_POWER_ITERATION
            std::vector<Matrix3> gen_covar_tans;
            gen_covar_tans.resize(pos_size, Matrix3({}, {}, {}));
        #endif

			for(size_t pos_idx = hdr.PositionStart, tc_idx = hdr.TexCoordStart, gen_idx = 0; gen_idx < pos_size;)
			{
				auto pos_prev_idx = pos_ind[pos_idx++];
				auto tc_prev_idx = tc_ind[tc_idx++];
				auto gen_prev_idx = gen_inds[gen_idx++];

				auto pos_current_idx = pos_ind[pos_idx++];
				auto tc_current_idx = tc_ind[tc_idx++];
				auto gen_current_idx = gen_inds[gen_idx++];

				auto pos_next_idx = pos_ind[pos_idx++];
				auto tc_next_idx = tc_ind[tc_idx++];
				auto gen_next_idx = gen_inds[gen_idx++];

				auto& pos_prev = pos[pos_prev_idx];
				auto& pos_current = pos[pos_current_idx];
				auto& pos_next = pos[pos_next_idx];
				
				auto& tc_prev = tc[tc_prev_idx];
				auto& tc_current = tc[tc_current_idx];
				auto& tc_next = tc[tc_next_idx];

				Vector3 tangent, binormal;

				GenerateTangentSpace(pos_prev, pos_current, pos_next,
                                     tc_prev, tc_current, tc_next,
                                     &tangent, &binormal);

                TGE_ASSERT(tangent.x || tangent.y || tangent.z &&
                           (std::isfinite(tangent.x) && std::isfinite(tangent.y) && std::isfinite(tangent.z)), "Invalid tangent");
                
                NormalizeSelf(&tangent);

            #ifdef TANGENT_POWER_ITERATION
                auto tangent_cov = Tempest::OuterProduct(tangent, tangent);
                
                gen_covar_tans[gen_prev_idx] += tangent_cov;
				gen_covar_tans[gen_current_idx] += tangent_cov;
				gen_covar_tans[gen_next_idx] += tangent_cov;

                gen_tans[gen_prev_idx] = tangent;
				gen_tans[gen_current_idx] = tangent;
				gen_tans[gen_next_idx] = tangent;
            #else
                gen_tans[gen_prev_idx] += tangent;
				gen_tans[gen_current_idx] += tangent;
				gen_tans[gen_next_idx] += tangent;
            #endif
			}

        #ifdef TANGENT_POWER_ITERATION
			for(uint32_t idx = 0; idx < pos_size; ++idx)
			{
				Vector3 tangent = gen_tans[idx];
                auto& covar = gen_covar_tans[idx];
                for(uint32_t iter = 0; iter < 16; ++iter)
                {
                    tangent = Normalize(covar.transform(tangent));
                }
                gen_tans[idx] = tangent;

                TGE_ASSERT(tangent.x || tangent.y || tangent.z, "Invalid tangent");
			}
        #else
            for(auto& tangent : gen_tans)
			{
                TGE_ASSERT(tangent.x || tangent.y || tangent.z &&
                           (std::isfinite(tangent.x) && std::isfinite(tangent.y) && std::isfinite(tangent.z)), "Invalid tangent");

				NormalizeSelf(&tangent);
			}
        #endif

			verts[num] = reinterpret_cast<const char*>(&gen_tans.front());
			inds[num] = &gen_inds.front();
			*stride += strides[num++] = sizeof(gen_tans.front());
		}
    }
    
    const Vector3* norms_vec;
	const int32_t* norm_indices;
	uint32_t norm_start = hdr.NormalStart;
    if(norm_size != 0)
    {
        norms_vec = &norm.front();
        norm_indices = &norm_ind[hdr.NormalStart];
        TGE_ASSERT(ind_count == 0 || ind_count == norm_size, "Indices should be the same"); ind_count = norm_size;
    }
    else
    {
		norm_start = 0;
        TGE_ASSERT((pos_size % 3) == 0, "Position indices should be multiple of 3");
       
        gen_norms.resize(max_ind - min_ind + 1, Vector3{0.0f, 0.0f, 0.0f});

        for(size_t pos_idx = hdr.PositionStart, gen_idx = 0; gen_idx < pos_size;)
        {
            auto pos_prev_idx = pos_ind[pos_idx++];
            auto gen_prev_idx = gen_inds[gen_idx++];

            auto pos_current_idx = pos_ind[pos_idx++];
            auto gen_current_idx = gen_inds[gen_idx++];

            auto pos_next_idx = pos_ind[pos_idx++];
            auto gen_next_idx = gen_inds[gen_idx++];

            auto& prev = pos[pos_prev_idx];
            auto& current = pos[pos_current_idx];
            auto& next = pos[pos_next_idx];
            auto d0 = prev - current;
            auto d1 = next - current;
            Vector3 norm = Cross(d1, d0);
            gen_norms[gen_prev_idx] += norm;
            gen_norms[gen_current_idx] += norm;
            gen_norms[gen_next_idx] += norm;
        }
        
        for(auto& norm : gen_norms)
        {
            NormalizeSelf(&norm);
        }
     
        norm_size = gen_norms.size();
        norms_vec = &gen_norms.front();
        norm_indices = &gen_inds.front();
    }

	if(flags & TEMPEST_OBJ_LOADER_GENERATE_CONSISTENT_NORMALS)
	{
		int32_t norm_min_ind = std::numeric_limits<int32_t>::max(), norm_max_ind = 0;
		for(size_t i = norm_start, iend = norm_start + norm_size; i < iend; ++i)
		{
			auto ind = norm_ind[i];
			if(norm_min_ind > ind)
				norm_min_ind = ind;
			if(norm_max_ind < ind)
				norm_max_ind = ind;
		}

		gen_coefs.resize(norm_max_ind - norm_min_ind + 1, 1.0f);
		gen_coefs_inds.resize(norm_size);

        for(size_t norm_idx = norm_start, gen_idx = 0; gen_idx < norm_size;)
        {
			auto norm_prev_idx = norm_ind[norm_idx++];
            auto norm_current_idx = norm_ind[norm_idx++];
            auto norm_next_idx = norm_ind[norm_idx++];

			auto& prev_norm = norms_vec[norm_prev_idx];
			auto& current_norm = norms_vec[norm_current_idx];
			auto& next_norm = norms_vec[norm_next_idx];

			float cos_prev_current = Dot(prev_norm, current_norm);
			float cos_next_current = Dot(next_norm, current_norm);
			float cos_prev_next = Dot(prev_norm, next_norm);

			float lowest_cos = Minf(cos_prev_current, cos_next_current);
			if(cos_prev_next < lowest_cos)
				lowest_cos = cos_prev_next;

			TGE_ASSERT(lowest_cos != 0.0f, "Ortho norms are bound to fail");

			auto gen_prev_idx = gen_coefs_inds[gen_idx++] = norm_prev_idx - norm_min_ind;
            auto gen_current_idx = gen_coefs_inds[gen_idx++] = norm_current_idx - norm_min_ind;
            auto gen_next_idx = gen_coefs_inds[gen_idx++] = norm_next_idx - norm_min_ind;

			auto& gen_prev_sdot = gen_coefs[gen_prev_idx];
			auto& gen_current_sdot = gen_coefs[gen_current_idx];
			auto& gen_next_sdot = gen_coefs[gen_next_idx];

			if(lowest_cos < gen_prev_sdot)
				gen_prev_sdot = lowest_cos;
			if(lowest_cos < gen_current_sdot)
				gen_current_sdot = lowest_cos;
			if(lowest_cos < gen_next_sdot)
				gen_next_sdot = lowest_cos;
        }

		float w_scale = 0.03632f; // Look the appendix of Reshetov et al. paper for more information 

		for(auto& gen_coef : gen_coefs)
        {
			float one_minus_cos = (1 - gen_coef);
			gen_coef = acosf(gen_coef)*(1.0f + w_scale*one_minus_cos*one_minus_cos);
        }

		verts[num] = reinterpret_cast<const char*>(&gen_coefs.front());
        inds[num] = &gen_coefs_inds.front();
        *stride += strides[num++] = sizeof(gen_coefs.front());
	}

	verts[num] = reinterpret_cast<const char*>(norms_vec);
    inds[num] = norm_indices;
    *stride += strides[num++] = sizeof(*norms_vec);

    std::vector<int32_t> interm_indices;
    if(num == 0)
        return;

    size_t data_start = res_data->size();
    InterleaveVertices(verts, strides, num, inds, ind_count, &interm_indices, res_data);

    if(!gen_tans.empty())
    {
        uint32_t tangent_offset = sizeof(Vector3) + sizeof(Vector2);
        uint32_t cur_stride = *stride;
        uint32_t normal_offset = cur_stride - sizeof(Vector3);

        for(char* data_iter = &(*res_data)[data_start], *data_iter_end = data_iter + res_data->size();
            data_iter < data_iter_end; data_iter += cur_stride)
        {
            Vector3* tangent = reinterpret_cast<Vector3*>(data_iter + tangent_offset);
            Vector3* normal = reinterpret_cast<Vector3*>(data_iter + normal_offset);

            TGE_ASSERT((tangent->x || tangent->y || tangent->z) &&
                       (std::isfinite(tangent->x) && std::isfinite(tangent->y) && std::isfinite(tangent->z)), "Broken tangent");

            *tangent = Normalize(*tangent - *normal*Dot(*normal, *tangent));
        }
    }

	if(res_inds)
	{
		if(interm_indices.size() < std::numeric_limits<uint16_t>::max())
		{
			size_t i = 0;
			size_t start_offset = res_inds->size();
			res_inds->resize(start_offset + interm_indices.size());
			for(size_t i = 0, iend = interm_indices.size(); i < iend; ++i)
			{
				auto ind = interm_indices[i];
				TGE_ASSERT(ind < std::numeric_limits<uint16_t>::max(), "Invalid index");
				(*res_inds)[start_offset + i] = static_cast<uint16_t>(ind);
			}
		}
		else
		{
			TGE_ASSERT(false, "Mesh splitting is unsupported");
		}
	}

    if(res_inds32)
    {
        res_inds32->insert(res_inds32->end(), interm_indices.begin(), interm_indices.end());
    }
}

template<class TBackend, class TShaderProgram, class TDrawBatch, class TStateObject, class TResourceTable>
bool LoadObjFileStaticGeometry(const std::string& filename, FileLoader* loader,
                               TShaderProgram** progs, TBackend* backend,
                               TextureTable<TBackend>* tex_table,
                               uint32_t* batch_count, TDrawBatch** batches,
                               uint32_t* state_count, TStateObject*** states,
                               TResourceTable*** res_tbl,
                               std::vector<char>* res_data_ptr, std::vector<int32_t>* res_ind_ptr)
{
    ObjLoader::Driver obj_loader_driver(Path(filename).directoryPath(), loader);
    if(loader)
    {
        auto* file_descr = loader->loadFileContent(filename);
        CreateAtScopeExit([loader, file_descr]() { loader->freeFileContent(file_descr); });
        obj_loader_driver.parseString(file_descr->Content, file_descr->ContentSize, filename);
    }
    else
    {
        auto parse_ret = obj_loader_driver.parseFile(filename);
        if(!parse_ret)
        {
            std::stringstream ss;
            ss << "The application has failed to parse an object file (refer to the error log for more information): " << filename << std::endl;
            Log(LogLevel::Error, ss.str());
            TGE_ASSERT(parse_ret, ss.str());
            return false;
        }
    }
    
    std::vector<uint16_t> res_inds;
    std::vector<char>   res_data;
    
    auto& groups = obj_loader_driver.getGroups();
    
    *batch_count = (uint32_t)groups.size();
    *batches = new TDrawBatch[groups.size()];
    
    size_t prev_ind_size = 0, prev_vert_size = 0;

    auto rt_fmt = Tempest::DataFormat::RGBA8;
    
    struct StateDescription
    {
        size_t        model;
        TStateObject* state;

        StateDescription(size_t _model, TStateObject* _state)
            :   model(_model), state(std::move(_state)) {}
    };
    std::vector<StateDescription> pstates;
    auto cleanup = CreateAtScopeExit([&pstates, backend] { for(auto& pstate : pstates) backend->destroyRenderResource(pstate.state); });

    auto& pos_ind = obj_loader_driver.getPositionIndices();
    auto& tc_ind = obj_loader_driver.getTexCoordIndices();
    auto& norm_ind = obj_loader_driver.getNormalIndices();
    
    size_t group_count = groups.size();

    auto gen_res_tbl = CreateScoped<TResourceTable**>(progs ? new TResourceTable*[group_count]{} : nullptr, [group_count](TResourceTable** res_tbl)
    {
        if(res_tbl)
        {
            for(size_t i = 0, iend = group_count; i < iend; ++i)
                delete res_tbl[i];
            delete[] res_tbl;
        }
    });

    auto base_dir = Path(filename).directoryPath() + "/";

    for(size_t i = 0, iend = group_count; i < iend; ++i)
    {
        size_t pos_size,
               tc_size,
               norm_size;
        if(iend - 1 == i)
        {
            pos_size = pos_ind.size() - groups.back().PositionStart;
            tc_size = tc_ind.size() - groups.back().TexCoordStart;
            norm_size = norm_ind.size() - groups.back().NormalStart;
        }
        else
        {
            pos_size = groups[i + 1].PositionStart - groups[i].PositionStart;
            tc_size = groups[i + 1].TexCoordStart - groups[i].TexCoordStart;
            norm_size = groups[i + 1].NormalStart - groups[i].NormalStart;
        }
        
        uint32_t vb_stride;

        InterleaveInterm(obj_loader_driver, groups[i], pos_size, tc_size, norm_size, &res_inds, &res_data, &vb_stride, res_ind_ptr);
        
        auto& batch = (*batches)[i];
        batch.VertexCount   = static_cast<uint32_t>(res_inds.size() - prev_ind_size);
        batch.BaseIndex     = static_cast<uint32_t>(prev_ind_size);
        batch.BaseVertex    = 0;
        batch.SortKey       = 0; // This could be regenerated on the fly
        batch.VertexBuffers[0].Offset = static_cast<uint32_t>(prev_vert_size);
        batch.VertexBuffers[0].Stride = vb_stride;

        if(progs)
        {
            auto& material = obj_loader_driver.getMaterials().at(groups[i].MaterialIndex);

            enum
            {
                AmbientAvailable = 1 << 0,
                SpecularAvailable = 1 << 1
            };

            size_t model = 0;
            size_t flags = 0;

            switch(material.IllumModel)
            {
            default: TGE_ASSERT(false, "Unsupported illumination model");
            case ObjMtlLoader::IlluminationModel::Diffuse: model = 0; break;
            case ObjMtlLoader::IlluminationModel::DiffuseAndAmbient: model = 1; flags = AmbientAvailable; break;
            case ObjMtlLoader::IlluminationModel::SpecularDiffuseAmbient: model = 2; flags = AmbientAvailable | SpecularAvailable;  break;
            }

            if(tc_size)
            {
                model += 3;
            }

            auto begin_state_iter = std::begin(pstates),
                end_state_iter = std::end(pstates);
            auto iter = std::find_if(begin_state_iter, end_state_iter, [&model](const StateDescription& st_desc) { return st_desc.model == model; });
            auto* shader_prog = progs[model];
            if(iter == end_state_iter)
            {
                DepthStencilStates ds_state;
                ds_state.DepthTestEnable = true;
                ds_state.DepthWriteEnable = true;
                auto pipeline_state = backend->createStateObject(&rt_fmt, 1, DataFormat::D24S8, shader_prog, DrawModes::TriangleList, nullptr, nullptr, &ds_state);
                batch.PipelineState = pipeline_state;
                pstates.emplace_back(model, pipeline_state);
            }
            else
            {
                batch.PipelineState = iter->state;
            }

            Matrix4 Imat;
            Imat.identity();

            auto& ambient = material.AmbientReflectivity;
            auto& diffuse = material.DiffuseReflectivity;
            auto& specular = material.SpecularReflectivity;

            auto globals_res_table = CreateResourceTable(shader_prog, "Globals", 1);
            TGE_ASSERT(globals_res_table, "Expecting valid table");
            globals_res_table->setResource("Globals.WorldViewProjectionTransform", Imat);
            globals_res_table->setResource("Globals.RotateTransform", Imat);
            if(flags & AmbientAvailable)
            {
				globals_res_table->setResource("Globals.AmbientDissolve", Vector4{ambient.x, ambient.y, ambient.z, material.Dissolve});
            }
			globals_res_table->setResource("Globals.DiffuseReflectivity", Vector4{diffuse.x, diffuse.y, diffuse.z, material.ReflectionSharpness});
            if(flags & SpecularAvailable)
            {
				globals_res_table->setResource("Globals.Specular", Vector4{specular.x, specular.y, specular.z, material.SpecularExponent});
            }

            if(tc_size)
            {
                if(!material.DiffuseReflectivityMap.empty())
                {
                    auto diff_map = tex_table->loadTexture(Path(base_dir + material.DiffuseReflectivityMap));
                    globals_res_table->setResource("Globals.DiffuseReflectivityMap", diff_map);
                }
                else
                {
					globals_res_table->setResource("Globals.DiffuseReflectivityMap", Vector4{0.0f, 0.0f, 0.0f, 0.0f});
                }
                if(flags & AmbientAvailable)
                {
                    if(!material.AmbientReflectivityMap.empty())
                    {
                        auto amb_map = tex_table->loadTexture(Path(base_dir + material.AmbientReflectivityMap));
                        globals_res_table->setResource("Globals.AmbientReflectivityMap", amb_map);
                    }
                    else
                    {
						globals_res_table->setResource("Globals.AmbientReflectivityMap", Vector4{0.0f, 0.0f, 0.0f, 0.0f});
                    }
                }
                if(flags & SpecularAvailable)
                {
                    if(!material.SpecularExponentMap.empty())
                    {
                        auto exp_map = tex_table->loadTexture(Path(base_dir + material.SpecularExponentMap));
                        globals_res_table->setResource("Globals.SpecularExponentMap", exp_map);
                    }
                    else
                    {
						globals_res_table->setResource("Globals.SpecularExponentMap", Vector4{0.0f, 0.0f, 0.0f, 0.0f});
                    }
                    if(!material.SpecularReflectivityMap.empty())
                    {
                        auto spec_map = tex_table->loadTexture(Path(base_dir + material.SpecularReflectivityMap));
                        globals_res_table->setResource("Globals.SpecularReflectivityMap", spec_map);
                    }
                    else
                    {
						globals_res_table->setResource("Globals.SpecularReflectivityMap", Vector4{0.0f, 0.0f, 0.0f, 0.0f});
                    }
                }
            }

            batch.ResourceTable = globals_res_table->getBakedTable();
            gen_res_tbl[i] = globals_res_table.release();
        }

        prev_ind_size = res_inds.size();
        prev_vert_size = res_data.size();
    }
    
    auto *vbo = backend->createBuffer(res_data.size(), ResourceBufferType::VertexBuffer, RESOURCE_STATIC_DRAW, &res_data.front());
    auto *ibo = backend->createBuffer(res_inds.size()*sizeof(uint16_t), ResourceBufferType::IndexBuffer, RESOURCE_STATIC_DRAW, &res_inds.front());
    
    for(size_t i = 0; i < *batch_count; ++i)
    {
        auto& batch = (*batches)[i];
        batch.IndexBuffer                   = ibo;
        batch.VertexBuffers[0].VertexBuffer = vbo;
    }
    
    *state_count = (uint32_t)pstates.size();
    *states = pstates.size() ? new TStateObject*[pstates.size()] : nullptr;
    for(size_t i = 0, iend = pstates.size(); i < iend; ++i)
    {
        (*states)[i] = pstates[i].state;
    }
    // Basically, stop the deletion process
    pstates.clear();

    *res_tbl = gen_res_tbl;
    gen_res_tbl = nullptr;

    if(res_data_ptr)
    {
        res_data_ptr->swap(res_data);
    }

    return true;
}

bool LoadObjFileStaticRTGeometry(const std::string& filename, FileLoader* loader, RTMeshBlob* rt_blob, uint32_t flags)
{
	ObjLoader::Driver obj_loader_driver(Path(filename).directoryPath(), loader);
	bool parse_ret;
    if(loader)
    {
        auto* file_descr = loader->loadFileContent(filename);
		if(file_descr)
		{
			auto at_exit = CreateAtScopeExit([loader, file_descr]() { loader->freeFileContent(file_descr); });
			parse_ret = obj_loader_driver.parseString(file_descr->Content, file_descr->ContentSize, filename);
		}
		else
		{
			parse_ret = false;
		}
    }
    else
    {
        parse_ret = obj_loader_driver.parseFile(filename);
	}

	if(!parse_ret)
    {
        std::stringstream ss;
        ss << "The application has failed to parse an object file (refer to the error log for more information): " << filename << std::endl;
        Log(LogLevel::Error, ss.str());
        TGE_ASSERT(parse_ret, ss.str());
        return false;
    }
    
    auto& groups = obj_loader_driver.getGroups();
    
    auto submesh_count = rt_blob->SubmeshCount = (uint32_t)groups.size();
    auto submeshes = rt_blob->Submeshes = new RTSubmesh[submesh_count];
    
    size_t prev_ind_size = 0, prev_vert_size = 0;

    auto rt_fmt = Tempest::DataFormat::RGBA8;
    
    auto& pos_ind = obj_loader_driver.getPositionIndices();
    auto& tc_ind = obj_loader_driver.getTexCoordIndices();
    auto& norm_ind = obj_loader_driver.getNormalIndices();
    
    for(auto& idx : pos_ind)
    {
        TGE_ASSERT(idx >= 0, "Invalid index");
    }

    for(auto& idx : tc_ind)
    {
        TGE_ASSERT(idx >= 0, "Invalid index");
    }

    for(auto& idx : norm_ind)
    {
        TGE_ASSERT(idx >= 0, "Invalid index");
    }
    
    size_t group_count = groups.size();

    auto base_dir = Path(filename).directoryPath() + "/";

	auto& material = obj_loader_driver.getMaterials();
	uint32_t material_count = static_cast<uint32_t>(material.size());
	rt_blob->MaterialCount = material_count;
	rt_blob->Materials = new RTMicrofacetMaterial[material_count];

	for(size_t material_idx = 0; material_idx < material_count; ++material_idx)
	{
		auto out_material = rt_blob->Materials + material_idx;

		auto& in_material = obj_loader_driver.getMaterials().at(material_idx);

		auto& ambient = in_material.AmbientReflectivity;
        auto& diffuse = in_material.DiffuseReflectivity;
        auto& specular = in_material.SpecularReflectivity;

		enum
        {
            AmbientAvailable = 1 << 0,
            SpecularAvailable = 1 << 1
        };

        size_t flags = 0;

        switch(in_material.IllumModel)
        {
        default: TGE_ASSERT(false, "Unsupported illumination model");
        case ObjMtlLoader::IlluminationModel::Diffuse: break;
        case ObjMtlLoader::IlluminationModel::DiffuseAndAmbient: flags = AmbientAvailable; break;
        case ObjMtlLoader::IlluminationModel::SpecularDiffuseAmbient: flags = AmbientAvailable | SpecularAvailable;  break;
        }

		/*
        if(flags & AmbientAvailable)
        {
			// ignore
        }
		*/

		
		out_material->Diffuse = Tempest::RGBToSpectrum(diffuse);

        if(flags & SpecularAvailable)
        {
			out_material->Specular = Tempest::RGBToSpectrum(specular);
			out_material->SpecularPower = { in_material.SpecularExponent, in_material.SpecularExponent };
        }
        else
        {
            out_material->Specular = {};
            out_material->SpecularPower = { 1.0f, 1.0f };
        }

        if(!in_material.DiffuseReflectivityMap.empty())
        {
			auto diff_map = Tempest::LoadImage(Path(base_dir + in_material.DiffuseReflectivityMap));
			if(diff_map)
			{
				rt_blob->TextureData.push_back(diff_map);
				out_material->DiffuseMap = diff_map;
			}
			else
			{
				Log(LogLevel::Error, "Failed to load diffuse component texture file: ", in_material.DiffuseReflectivityMap);
			}
        }
		/*
        if(flags & AmbientAvailable)
        {
            // ignore
        }
		*/
        if((flags & SpecularAvailable) && !in_material.SpecularExponentMap.empty() && !in_material.SpecularReflectivityMap.empty())
        {
            std::unique_ptr<Texture> exp_map(Tempest::LoadImage(Path(base_dir + in_material.SpecularExponentMap)));
			std::unique_ptr<Texture> spec_str_map(Tempest::LoadImage(Path(base_dir + in_material.SpecularReflectivityMap)));

			if(exp_map && spec_str_map)
			{
				auto& spec_str_hdr = spec_str_map->getHeader();
			
				TextureDescription tex_desc;
				tex_desc.Width = spec_str_hdr.Width;
				tex_desc.Height = spec_str_hdr.Height;
				uint32_t* data_ptr = new uint32_t[tex_desc.Width*tex_desc.Height];
				auto spec_map = new Texture(tex_desc, reinterpret_cast<uint8_t*>(data_ptr));
				rt_blob->TextureData.push_back(spec_map);

				for(uint32_t y = 0; y < tex_desc.Height; ++y)
				{
					for(uint32_t x = 0; x < tex_desc.Width; ++x)
					{
						Vector2 tc{ (float)x/(tex_desc.Width - 1), (float)y/(tex_desc.Height - 1) };
						auto color = spec_str_map->sampleRGB(tc);
						auto exp = exp_map->sampleRed(tc);
						data_ptr[y*tex_desc.Width + x] = ToColor({ color.x, color.y, color.z, exp }); 
					}
				}
			}
			else
			{
				if(!exp_map)
				{
					Log(LogLevel::Error, "Failed to load specular component texture file: ", in_material.SpecularExponentMap);
				}
				if(!spec_str_map)
				{
					Log(LogLevel::Error, "Failed to load specular component texture file: ", in_material.SpecularReflectivityMap);
				}
			}
        }

        if(IsZero(in_material.Emission))
        {
            out_material->Model = IlluminationModel::Emissive;
            out_material->Diffuse = in_material.Emission;
        }

		out_material->setup();
	}

    for(size_t submesh_idx = 0, submesh_count = group_count; submesh_idx < submesh_count; ++submesh_idx)
    {
		auto& submesh = rt_blob->Submeshes[submesh_idx];

        size_t pos_size,
               tc_size,
               norm_size;
        if(submesh_count - 1 == submesh_idx)
        {
            pos_size = pos_ind.size() - groups.back().PositionStart;
            tc_size = tc_ind.size() - groups.back().TexCoordStart;
            norm_size = norm_ind.size() - groups.back().NormalStart;
        }
        else
        {
            pos_size = groups[submesh_idx + 1].PositionStart - groups[submesh_idx].PositionStart;
            tc_size = groups[submesh_idx + 1].TexCoordStart - groups[submesh_idx].TexCoordStart;
            norm_size = groups[submesh_idx + 1].NormalStart - groups[submesh_idx].NormalStart;
        }
        
        uint32_t vb_stride;

        InterleaveInterm(obj_loader_driver, groups[submesh_idx], pos_size, tc_size, norm_size, nullptr, &rt_blob->VertexData, &vb_stride, &rt_blob->IndexData, flags);
        
		submesh.Material	  = &rt_blob->Materials[groups[submesh_idx].MaterialIndex];
        submesh.VertexCount   = static_cast<uint32_t>(rt_blob->IndexData.size() - prev_ind_size);
        submesh.BaseIndex     = static_cast<uint32_t>(prev_ind_size);
        submesh.VertexOffset  = static_cast<uint32_t>(prev_vert_size);
        submesh.Stride		  = vb_stride;
		
        prev_ind_size = rt_blob->IndexData.size();
        prev_vert_size = rt_blob->VertexData.size();
    }

    return true;
}

template bool LoadObjFileStaticGeometry(const std::string& filename, FileLoader* loader,
                                        GLShaderProgram** progs, GLRenderingBackend* backend,
                                        TextureTable<GLRenderingBackend>* tex_table,
                                        uint32_t* batch_count, GLDrawBatch** batches,
                                        uint32_t* num_states, GLStateObject*** states,
                                        GLResourceTable*** res_tbl,
                                        std::vector<char>* res_data_ptr, std::vector<int32_t>* res_ind_ptr);
}