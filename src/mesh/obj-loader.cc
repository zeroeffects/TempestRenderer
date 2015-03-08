/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2009 2010 2011 2012 Zdravko Velinov
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

namespace Tempest
{
class FileLoader;

static void InterleaveInterm(ObjLoader::Driver& obj_loader_driver, const ObjLoader::GroupHeader& hdr, size_t pos_size, size_t tc_size, size_t norm_size,
                             std::vector<uint16>* res_inds, std::vector<char>* res_data, uint32* stride)
{
    int32        strides[3];
    const int32* inds[3];
    const char*  verts[3];

    std::vector<Vector3> gen_norms;
    
    *stride = 0;

    size_t num = 0, ind_count = 0;
    auto& pos = obj_loader_driver.getPositions();
    auto& tc = obj_loader_driver.getTexCoords();
    auto& norm = obj_loader_driver.getNormals();
    
    auto& pos_ind = obj_loader_driver.getPositionIndices();
    auto& tc_ind = obj_loader_driver.getTexCoordIndices();
    auto& norm_ind = obj_loader_driver.getNormalIndices();
    if(pos_size != 0)
    {
        verts[num] = reinterpret_cast<const char*>(&pos.front());
        inds[num] = &pos_ind.front();
        *stride += strides[num++] = sizeof(pos.front());
        ind_count = pos_size;
    }
    
    if(tc_size != 0)
    {
        verts[num] = reinterpret_cast<const char*>(&tc.front());
        inds[num] = &tc_ind.front();
        *stride += strides[num++] = sizeof(tc.front());
        TGE_ASSERT(ind_count == 0 || ind_count == tc_size, "Indices should be the same"); ind_count = tc_size;
    }
    
    if(norm_size != 0)
    {
        verts[num] = reinterpret_cast<const char*>(&norm.front());
        inds[num] = &norm_ind.front();
        *stride += strides[num++] = sizeof(norm.front());
        TGE_ASSERT(ind_count == 0 || ind_count == norm_size, "Indices should be the same"); ind_count = norm_size;
    }
    else if(pos_size == 0)
    {
        return;
    }
    else
    {
        TGE_ASSERT(pos_ind.size() % 3, "Position indices should be multiple of 3");
        
        gen_norms.resize(pos.size(), Vector3(0.0f, 0.0f, 0.0f));
        for(size_t i = 0; i < pos_ind.size();)
        {
            auto prev_idx = pos_ind[i++];
            auto current_idx = pos_ind[i++];
            auto next_idx = pos_ind[i++];
            auto& prev = pos[prev_idx];
            auto& current = pos[current_idx];
            auto& next = pos[next_idx];
            auto current_v3 = ToVector3(current);
            auto d0 = ToVector3(prev) - current_v3;
            auto d1 = ToVector3(next) - current_v3;
            Vector3 norm = d0.cross(d1);
            gen_norms[prev_idx] += norm;
            gen_norms[current_idx] += norm;
            gen_norms[next_idx] += norm;
        }
        
        for(auto& norm : gen_norms)
        {
            norm.normalize();
        }
        
        verts[num] = reinterpret_cast<const char*>(&gen_norms.front());
        inds[num] = &pos_ind.front();
        *stride += strides[num++] = sizeof(gen_norms.front());
    }
    
    std::vector<int32> interm_indices;
    if(num != 0)
    {
        InterleaveVertices(verts, strides, num, inds, ind_count, &interm_indices, res_data);
    }

    if(interm_indices.size() < std::numeric_limits<uint16>::max())
    {
        size_t i = 0;
        res_inds->resize(interm_indices.size());
        for(size_t i = 0, iend = interm_indices.size(); i < iend; ++i)
        {
            auto ind = interm_indices[i];
            TGE_ASSERT(ind < std::numeric_limits<uint16>::max(), "Invalid index");
            (*res_inds)[i] = static_cast<uint16>(ind);
        }
    }
    else
    {
        TGE_ASSERT(false, "Mesh splitting is unsupported");
    }
}

template<class TBackend, class TShaderProgram, class TDrawBatch, class TStateObject>
bool LoadObjFileStaticGeometry(const string& filename, FileLoader* loader,
                               TShaderProgram** progs, TBackend* backend,
                               size_t* batch_count, TDrawBatch** batches,
                               size_t* state_count, TStateObject*** states)
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
    
    std::vector<uint16> res_inds;
    std::vector<char>   res_data;
    
    auto& groups = obj_loader_driver.getGroups();
    
    obj_loader_driver.normalizeIndices();

    *batch_count = groups.size();
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
    
    for(size_t i = 0, iend = groups.size(); i < iend; ++i)
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
        
        uint32 vb_stride;

        InterleaveInterm(obj_loader_driver, groups[i], pos_size, tc_size, norm_size, &res_inds, &res_data, &vb_stride);
        
        auto& batch = (*batches)[i];
        batch.VertexCount   = static_cast<uint32>(res_inds.size() - prev_ind_size);
        batch.BaseIndex     = static_cast<uint32>(prev_ind_size);
        batch.BaseVertex    = 0;
        batch.SortKey       = 0; // This could be regenerated on the fly
        batch.VertexBuffers[0].Offset = static_cast<uint32>(prev_vert_size);
        batch.VertexBuffers[0].Stride = vb_stride;

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
            auto pipeline_state = backend->createStateObject(&rt_fmt, 1, DataFormat::D24S8, shader_prog);
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
        globals_res_table->setResource("Globals.Transform", Imat);
        if(flags & AmbientAvailable)
        {
            globals_res_table->setResource("Globals.AmbientDissolve", Vector4(ambient.x(), ambient.y(), ambient.z(), material.Dissolve));
        }
        globals_res_table->setResource("Globals.DiffuseReflectivity", Vector4(diffuse.x(), diffuse.y(), diffuse.z(), material.ReflectionSharpness));
        if(flags & SpecularAvailable)
        {
            globals_res_table->setResource("Globals.Specular", Vector4(specular.x(), specular.y(), specular.z(), material.SpecularExponent));
        }
        
        batch.ResourceTable = globals_res_table->extractBakedTable();        
       
        prev_ind_size = res_inds.size();
        prev_vert_size = res_data.size()*batch.VertexBuffers[0].Stride;
    }
    
    auto *vbo = backend->createBuffer(res_data.size(), VBType::VertexBuffer, RESOURCE_STATIC_DRAW, &res_data.front());
    auto *ibo = backend->createBuffer(res_inds.size()*sizeof(uint16), VBType::IndexBuffer, RESOURCE_STATIC_DRAW, &res_inds.front());
    
    for(size_t i = 0; i < *batch_count; ++i)
    {
        auto& batch = (*batches)[i];
        batch.IndexBuffer                   = ibo;
        batch.VertexBuffers[0].VertexBuffer = vbo;
    }
    
    *state_count = pstates.size();
    *states = new TStateObject*[pstates.size()];
    for(size_t i = 0, iend = pstates.size(); i < iend; ++i)
    {
        (*states)[i] = pstates[i].state;
    }
    // Basically, stop the deletion process
    pstates.clear();

    return true;
}

template bool LoadObjFileStaticGeometry(const string& filename, FileLoader* loader,
                                        GLShaderProgram** progs, GLRenderingBackend* backend,
                                        size_t* batch_count, GLDrawBatch** batches,
                                        size_t* num_states, GLStateObject*** states);
}