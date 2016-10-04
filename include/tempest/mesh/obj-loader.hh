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

#ifndef _TEMPEST_OBJ_LOADER_HH_
#define _TEMPEST_OBJ_LOADER_HH_

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace Tempest
{
enum
{
	TEMPEST_OBJ_LOADER_GENERATE_TANGENTS = 1 << 0,
	TEMPEST_OBJ_LOADER_GENERATE_CONSISTENT_NORMALS = 1 << 1
};

template<class TBackend> class TextureTable;
class FileLoader;
struct RTSubmesh;
struct RTMicrofacetMaterial;
class Texture;

struct RTMeshBlob
{
    uint32_t			  SubmeshCount = 0;
    RTSubmesh*			  Submeshes = nullptr;
	std::vector<char>	  VertexData;
	std::vector<int32_t>  IndexData;
	std::vector<Texture*> TextureData;
	uint32_t			  Stride = 0;
	RTMicrofacetMaterial* Materials = nullptr;
	uint32_t			  MaterialCount = 0;

	~RTMeshBlob();
};

// Kind of horrible.
// TODO:
// - Create some not horrible material cache and use it while loading
// - Load stuff in structures
template<class TBackend, class TShaderProgram, class TDrawBatch, class TStateObject, class TResourceTable>
bool LoadObjFileStaticGeometry(const std::string& filename, FileLoader* loader,
                               TShaderProgram** progs, TBackend* backend,
                               TextureTable<TBackend>* tex_table,
                               uint32_t* batch_count, TDrawBatch** batches,
                               uint32_t* num_states, TStateObject*** states,
                               TResourceTable*** res_tbls,
                               std::vector<char>* res_data_ptr, std::vector<int32_t>* res_ind_ptr);

bool LoadObjFileStaticRTGeometry(const std::string& filename, FileLoader* loader, RTMeshBlob* rt_mesh, uint32_t flags = 0);

template<class TBackend>
struct GPUMeshBlob
{
    typedef typename TBackend::CommandBufferType::DrawBatchType DrawBatchType;
    typedef typename TBackend::ShaderProgramType::ResourceTableType ResourceTableType;
    TBackend*                            Backend;
    DrawBatchType*                       DrawBatches;
    typename TBackend::StateObjectType** StateObjects;
    uint32_t                             DrawBatchCount;
    uint32_t                             StateCount;
    ResourceTableType**                  ResourceTables;

    ~GPUMeshBlob()
    {
        Backend->destroyRenderResource(DrawBatches[0].VertexBuffers[0].VertexBuffer);
        Backend->destroyRenderResource(DrawBatches[0].IndexBuffer);
        if(StateCount)
        {
            for(uint32_t i = 0, iend = DrawBatchCount; i < iend; ++i)
            {
                delete ResourceTables[i];
            }
            delete[] ResourceTables;
            for(uint32_t i = 0, iend = StateCount; i < iend; ++i)
            {
                Backend->destroyRenderResource(StateObjects[i]);
            }
            delete[] StateObjects;
        }
        delete[] DrawBatches;
    }
};

template<class TBackend>
struct MixedMeshBlob: public GPUMeshBlob<TBackend>
{
    std::vector<char>                   Data;
    std::vector<int32_t>                Indices;
};

template<class TBackend>
std::unique_ptr<GPUMeshBlob<TBackend>> LoadObjFileStaticGeometryGPUBlob(const std::string& filename, FileLoader* loader,
                                                                     typename TBackend::ShaderProgramType** progs, TextureTable<TBackend>* tex_table, TBackend* backend)
{
    std::unique_ptr<GPUMeshBlob<TBackend>> result(new GPUMeshBlob<TBackend>);
    result->Backend = backend;
    auto status = LoadObjFileStaticGeometry(filename, loader, progs, backend, tex_table,
                                            &result->DrawBatchCount, &result->DrawBatches,
                                            &result->StateCount, &result->StateObjects,
                                            &result->ResourceTables,
                                            nullptr, nullptr);
    return status ? std::move(result) : std::unique_ptr<GPUMeshBlob<TBackend>>();
}

template<class TBackend>
std::unique_ptr<MixedMeshBlob<TBackend>> LoadObjFileStaticGeometryMixedBlob(const std::string& filename, FileLoader* loader,
                                                                       typename TBackend::ShaderProgramType** progs, TextureTable<TBackend>* tex_table, TBackend* backend)
{
    std::unique_ptr<MixedMeshBlob<TBackend>> result(new MixedMeshBlob<TBackend>);
    result->Backend = backend;
    auto status = LoadObjFileStaticGeometry(filename, loader, progs, backend, tex_table,
                                            &result->DrawBatchCount, &result->DrawBatches,
                                            &result->StateCount, &result->StateObjects,
                                            &result->ResourceTables,
                                            &result->Data, &result->Indices);
    return status ? std::move(result) : std::unique_ptr<MixedMeshBlob<TBackend>>();
}
}

#endif // _TEMPEST_OBJ_LOADER_HH_