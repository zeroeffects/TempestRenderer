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

#include <fstream>

namespace Tempest
{
// Kind of horrible. TODO: Create some not horrible material cache and use it while loading
template<class TBackend, class TShaderProgram, class TDrawBatch, class TStateObject, class TResourceTable>
bool LoadObjFileStaticGeometry(const string& filename, FileLoader* loader,
                               TShaderProgram** progs, TBackend* backend,
                               TextureTable<TBackend>* tex_table,
                               size_t* batch_count, TDrawBatch** batches,
                               size_t* num_states, TStateObject*** states,
                               TResourceTable*** res_tbls);

template<class TBackend>
struct MeshBlob
{
    typedef typename TBackend::CommandBufferType::DrawBatchType DrawBatchType;
    typedef typename TBackend::ShaderProgramType::ResourceTableType ResourceTableType;
    TBackend*                            Backend;
    DrawBatchType*                       DrawBatches;
    typename TBackend::StateObjectType** StateObjects;
    size_t                               DrawBatchCount;
    size_t                               StateCount;
    ResourceTableType**                  ResourceTables;

    ~MeshBlob()
    {
        Backend->destroyRenderResource(DrawBatches[0].VertexBuffers[0].VertexBuffer);
        Backend->destroyRenderResource(DrawBatches[0].IndexBuffer);
        for(size_t i = 0, iend = DrawBatchCount; i < iend; ++i)
        {
            delete ResourceTables[i];
        }
        delete[] ResourceTables;
        delete[] DrawBatches;
        for(size_t i = 0, iend = StateCount; i < iend; ++i)
        {
            Backend->destroyRenderResource(StateObjects[i]);
        }
        delete[] StateObjects;
    }
};

template<class TBackend>
std::unique_ptr<MeshBlob<TBackend>> LoadObjFileStaticGeometryBlob(const string& filename, FileLoader* loader,
                                                                  typename TBackend::ShaderProgramType** progs, TextureTable<TBackend>* tex_table, TBackend* backend)
{
    std::unique_ptr<MeshBlob<TBackend>> result(new MeshBlob<TBackend>);
    result->Backend = backend;
    auto status = LoadObjFileStaticGeometry(filename, loader, progs, backend, tex_table,
                                            &result->DrawBatchCount, &result->DrawBatches,
                                            &result->StateCount, &result->StateObjects,
                                            &result->ResourceTables);
    return status ? std::move(result) : std::unique_ptr<MeshBlob<TBackend>>();
}
}