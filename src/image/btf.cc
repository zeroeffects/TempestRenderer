/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2016 Zdravko Velinov
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

#include "tempest/image/btf.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/threads.hh"
#include "tempest/math/sampling3.hh"

#ifndef DISABLE_CUDA
#   include <cuda_runtime_api.h>
#endif

#include <algorithm>

namespace Tempest
{
void DestroyCPUBTF(BTF* btf)
{
	delete[] btf->HeightMap;
	btf->HeightMap = nullptr;
	delete[] btf->LeftSingularU;
	btf->LeftSingularU = nullptr;
	delete[] btf->RightSingularSxV;
	btf->RightSingularSxV = nullptr;
	delete[] btf->LightsParabolic;
	btf->LightsParabolic = nullptr;
	delete[] btf->LightBVH;
	btf->LightBVH = nullptr;
	delete[] btf->Offsets;
	btf->Offsets = nullptr;
	delete[] btf->ComponentCounts;
	btf->ComponentCounts = nullptr;
    free(btf->LightIndices);
	btf->LightIndices = nullptr;
    delete[] btf->Edges;
    btf->Edges = nullptr;

	/*
		delete[] btf->ViewBVH;
		btf->ViewBVH = nullptr;
        free(btf->ViewIndices);
		btf->ViewIndices = nullptr;
	*/
}

#ifndef DISABLE_CUDA
void DestroyGPUBTF(BTF* btf)
{
	cudaFree(btf);
}

BTF* CutBTF(const BTF* source_btf, uint32_t start_x, uint32_t start_y, uint32_t width, uint32_t height)
{
    if(start_x + width > source_btf->Width ||
       start_y + height > source_btf->Height ||
       width == 0 || height == 0)
        return nullptr;

    BTFPtr cut_btf(new BTF);
    *cut_btf = *source_btf;
    if(cut_btf->HeightMap)
    {
        auto height_map_area = (size_t)source_btf->HeightMapSize.Width*source_btf->HeightMapSize.Height;
        cut_btf->HeightMap = new uint16_t[height_map_area];
        std::copy_n(source_btf->HeightMap, height_map_area, cut_btf->HeightMap);
    }

    auto light_count = source_btf->LightCount;
    cut_btf->LightsParabolic = new Vector2[light_count];
    std::copy_n(source_btf->LightsParabolic, light_count, cut_btf->LightsParabolic);

    size_t light_bvh_size = (2*source_btf->LightTriangleCount - 1);
    cut_btf->LightBVH = new SimpleStacklessLBVH2Node<AABB2>[light_bvh_size];
    std::copy_n(source_btf->LightBVH, light_bvh_size, cut_btf->LightBVH);

    size_t offsets_size = 2*source_btf->ChannelCount;
    cut_btf->Offsets = new uint32_t[offsets_size];
    std::copy_n(source_btf->Offsets, offsets_size, cut_btf->Offsets);

    cut_btf->ComponentCounts = new uint32_t[source_btf->ChannelCount];
    std::copy_n(source_btf->ComponentCounts, source_btf->ChannelCount, cut_btf->ComponentCounts);

    size_t light_indices_size = 3*source_btf->LightTriangleCount;
    cut_btf->LightIndices = new uint32_t[light_indices_size];
    std::copy_n(source_btf->LightIndices, light_indices_size, cut_btf->LightIndices);

    cut_btf->Edges = new Edge[source_btf->EdgeCount];
    std::copy_n(source_btf->Edges, source_btf->EdgeCount, cut_btf->Edges);

    cut_btf->Width = width;
    cut_btf->Height = height;

    auto row_count = light_count*light_count;
    auto src_column_count = source_btf->Width*source_btf->Height;
    auto dst_column_count = cut_btf->Width*cut_btf->Height;

    cut_btf->LeftSingularU = new uint8_t[cut_btf->LeftSingularUSize];
    cut_btf->RightSingularSxVSize = dst_column_count*(source_btf->RightSingularSxVSize/src_column_count);
	cut_btf->RightSingularSxV = new uint8_t[cut_btf->RightSingularSxVSize];
	
    uint32_t prev_orig_u_end = 0, prev_orig_sxv_end = 0,
             prev_u_end = 0, prev_sxv_end = 0;
    uint32_t chan_count = cut_btf->ChannelCount;
    for(uint32_t channel_idx = 0; channel_idx < chan_count; ++channel_idx)
    {
        auto component_count = source_btf->ComponentCounts[channel_idx];
        auto src_u_offset = source_btf->Offsets[channel_idx];
        uint32_t u_offset = src_u_offset;
        if(src_u_offset == prev_orig_u_end)
        {
            u_offset = prev_u_end;
        }

        cut_btf->Offsets[channel_idx] = u_offset;

        auto src_sxv_offset = source_btf->Offsets[chan_count + channel_idx];
        uint32_t sxv_offset = src_sxv_offset;
        if(sxv_offset == prev_orig_sxv_end)
        {
            sxv_offset = prev_sxv_end;
        }

        cut_btf->Offsets[chan_count + channel_idx] = sxv_offset;

        for(uint32_t y = 0, yend = height; y < yend; ++y)
            for(uint32_t x = 0, xend = width; x < xend; ++x)
            {        
                uint32_t src_SxV_elem_offset = ((y + start_y)*source_btf->Width + x + start_x)*component_count*source_btf->SxVElementStride;
		        auto src_SxVslice = source_btf->RightSingularSxV + src_sxv_offset + src_SxV_elem_offset;

                uint32_t dst_SxV_elem_offset = (y*width + x)*component_count*cut_btf->SxVElementStride;
                auto dst_SxVSlice = cut_btf->RightSingularSxV + sxv_offset + dst_SxV_elem_offset;

                std::copy_n(src_SxVslice, component_count*cut_btf->SxVElementStride, dst_SxVSlice);
            }

        for(uint32_t lv_idx = 0, lv_end = source_btf->LightCount*source_btf->LightCount; lv_idx < lv_end; ++lv_idx)
        {
            uint32_t u_elem_offset = lv_idx*component_count*source_btf->UElementStride;
		    auto* src_Uslice = source_btf->LeftSingularU + src_u_offset + u_elem_offset;
            auto* dst_Uslice = cut_btf->LeftSingularU + u_offset + u_elem_offset;

            std::copy_n(src_Uslice, component_count*source_btf->UElementStride, dst_Uslice);
        }

        prev_orig_u_end = src_u_offset + row_count*component_count*source_btf->UElementStride;
        prev_orig_sxv_end = src_sxv_offset + src_column_count*component_count*source_btf->SxVElementStride;

        prev_u_end = u_offset + row_count*component_count*cut_btf->UElementStride;
        prev_sxv_end = sxv_offset + dst_column_count*component_count*cut_btf->SxVElementStride;
    }

    return cut_btf.release();
}

BTF* CreateGPUBTF(BTF* cpu_btf)
{
	uint32_t total_component_count = cpu_btf->ComponentCounts[0];
	for(uint32_t idx = 1; idx < cpu_btf->ChannelCount; ++idx)
	{
		total_component_count += cpu_btf->ComponentCounts[idx];
	}
	size_t light_parabolic_size = cpu_btf->LightCount*sizeof(cpu_btf->LightsParabolic[0]);
	size_t light_bvh_size = (2*cpu_btf->LightTriangleCount - 1)*sizeof(cpu_btf->LightBVH[0]);
	size_t light_indices_size = 3*cpu_btf->LightTriangleCount*sizeof(cpu_btf->LightIndices[0]);
	size_t offsets_size = 2*cpu_btf->ChannelCount*sizeof(cpu_btf->Offsets[0]);
	size_t component_counts_size = cpu_btf->ChannelCount*sizeof(cpu_btf->ComponentCounts[0]);

	size_t left_singular_size = cpu_btf->LeftSingularUSize;
	size_t right_singular_size = cpu_btf->RightSingularSxVSize;

	size_t btf_header_aligned_size = AlignAddress(sizeof(BTF), sizeof(size_t));
	size_t left_singular_aligned_size = AlignAddress(cpu_btf->LeftSingularUSize, sizeof(size_t));
	size_t right_singular_aligned_size = AlignAddress(cpu_btf->RightSingularSxVSize, sizeof(size_t));
	size_t light_parabolic_aligned_size = AlignAddress(light_parabolic_size, sizeof(size_t));
	size_t light_bvh_aligned_size = AlignAddress(light_bvh_size, sizeof(size_t));
    size_t edges_aligned_size = AlignAddress(cpu_btf->EdgeCount*sizeof(Edge), sizeof(size_t));
	size_t light_indices_aligned_size = AlignAddress(light_indices_size, sizeof(size_t));
	size_t offsets_aligned_size = AlignAddress(offsets_size, sizeof(size_t));
	size_t component_counts_aligned_size = AlignAddress(component_counts_size, sizeof(size_t));

	size_t total_size = btf_header_aligned_size + left_singular_aligned_size + right_singular_aligned_size + light_parabolic_aligned_size +
						light_bvh_aligned_size + edges_aligned_size + light_indices_aligned_size + offsets_aligned_size + component_counts_aligned_size;

	auto btf_data = CREATE_SCOPED(uint8_t*, ::cudaFree);
	auto status = cudaMalloc(reinterpret_cast<void**>(&btf_data), total_size);
	if(status != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to allocate GPU BTF");
		return nullptr;
	}

	uint8_t* data_iter = btf_data.get();

	uint8_t* gpu_BTF = data_iter;
	data_iter += btf_header_aligned_size;
	uint8_t* gpu_LeftSingularU = data_iter;
	data_iter += left_singular_aligned_size;
	uint8_t* gpu_RightSingularSxV = data_iter;
	data_iter += right_singular_aligned_size;
	uint8_t* gpu_LightsParabolic = data_iter;
	data_iter += light_parabolic_aligned_size;
	uint8_t* gpu_LightBVH = data_iter;
	data_iter += light_bvh_aligned_size;
    uint8_t* gpu_Edges = data_iter;
    data_iter += edges_aligned_size;
	uint8_t* gpu_LightIndices = data_iter;
	data_iter += light_indices_aligned_size;
	uint8_t* gpu_Offsets = data_iter;
	data_iter += offsets_aligned_size;
	uint8_t* gpu_Components = data_iter;
	data_iter += component_counts_aligned_size;

	TGE_ASSERT(data_iter - btf_data == total_size, "Invalid btf data population");

	BTF btf_cpu_tmp = *cpu_btf;
	btf_cpu_tmp.LeftSingularU = gpu_LeftSingularU;
	btf_cpu_tmp.RightSingularSxV = gpu_RightSingularSxV;
	btf_cpu_tmp.LightsParabolic = reinterpret_cast<Vector2*>(gpu_LightsParabolic);
	btf_cpu_tmp.LightBVH = reinterpret_cast<SimpleStacklessLBVH2Node<AABB2>*>(gpu_LightBVH);
    btf_cpu_tmp.Edges = reinterpret_cast<Edge*>(gpu_Edges);
    btf_cpu_tmp.LightIndices = reinterpret_cast<uint32_t*>(gpu_LightIndices);
	btf_cpu_tmp.Offsets = reinterpret_cast<uint32_t*>(gpu_Offsets);
	btf_cpu_tmp.ComponentCounts = reinterpret_cast<uint32_t*>(gpu_Components);

	/*
		btf_cpu_tmp.ViewBVH = gpu_ViewBVH;
	    btf_cpu_tmp.ViewIndices = gpu_ViewIndices;
	*/

	status = cudaMemcpy(gpu_BTF, &btf_cpu_tmp, sizeof(btf_cpu_tmp), cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy BTF header");
		return nullptr;
	}

	status = cudaMemcpy(gpu_LeftSingularU, cpu_btf->LeftSingularU, left_singular_size, cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy U matrix");
		return nullptr;
	}

	status = cudaMemcpy(gpu_RightSingularSxV, cpu_btf->RightSingularSxV, right_singular_size, cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy SxV matrix");
		return nullptr;
	}

	status = cudaMemcpy(gpu_LightsParabolic, cpu_btf->LightsParabolic, light_parabolic_size, cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy Light data in parabolic coordinates");
		return nullptr;
	}

	status = cudaMemcpy(gpu_LightBVH, cpu_btf->LightBVH, light_bvh_size, cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy Light BVH");
		return nullptr;
	}

    status = cudaMemcpy(gpu_Edges, cpu_btf->Edges, edges_aligned_size, cudaMemcpyHostToDevice);
    if(status != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to copy edges information");
        return nullptr;
    }

    status = cudaMemcpy(gpu_LightIndices, cpu_btf->LightIndices, light_indices_size, cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy Light indices");
		return nullptr;
	}

	status = cudaMemcpy(gpu_Offsets, cpu_btf->Offsets, offsets_size, cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy offsets");
		return nullptr;
	}

	status = cudaMemcpy(gpu_Components, cpu_btf->ComponentCounts, component_counts_size, cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to copy components");
		return nullptr;
	}

	/*
		status = cudaMemcpy(gpu_ViewBVH, cpu_btf->ViewBVH, view_bvh_size, cudaMemcpyHostToDevice);
		if(status != cudaSuccess)
		{
			Log(LogLevel::Error, "Failed to copy View BVH");
			return nullptr;
		}
	    status = cudaMemcpy(gpu_ViewIndices, cpu_btf->ViewIndices, view_indices_size, cudaMemcpyHostToDevice);
		if(status != cudaSuccess)
		{
			Log(LogLevel::Error, "Failed to copy View indices");
			return nullptr;
		}
	*/

	return reinterpret_cast<BTF*>(btf_data.release());
}
#endif

BTF* CreateDummyBTF(uint32_t light_count)
{
    unsigned seed = 123;

    Tempest::BTFPtr zeroth_btf(new Tempest::BTF);
	zeroth_btf->ConsineFlag = false;
	zeroth_btf->ChannelCount = 3;
	zeroth_btf->Width = 1,
	zeroth_btf->Height = 1;
	zeroth_btf->DynamicRangeReduction = false;
	zeroth_btf->ColorModel = 0;
	zeroth_btf->DataSize = sizeof(float);
	zeroth_btf->RowCount = light_count*light_count*zeroth_btf->ChannelCount;
	zeroth_btf->ColumnCount = 1;
	auto left_singular_size = zeroth_btf->LeftSingularUSize = zeroth_btf->RowCount*zeroth_btf->DataSize;
    auto right_singular_size = zeroth_btf->RightSingularSxVSize = zeroth_btf->DataSize;
    zeroth_btf->LeftSingularU = new uint8_t[left_singular_size];
    zeroth_btf->RightSingularSxV = new uint8_t[right_singular_size];

    *reinterpret_cast<float*>(zeroth_btf->RightSingularSxV) = 1.0f;

    zeroth_btf->LightsParabolic = new Tempest::Vector2[light_count];
		
	uint32_t sqrt_light_count = (uint32_t)sqrtf((float)light_count);
	TGE_ASSERT((light_count % sqrt_light_count) == 0, "Invalid sample count");

	for(uint32_t b = 0; b < sqrt_light_count; ++b)
		for(uint32_t a = 0; a < sqrt_light_count; ++a)
		{
			auto dir = Tempest::UniformSampleHemisphere((float)(a + Tempest::FastFloatRand(seed))/sqrt_light_count, (float)(b + Tempest::FastFloatRand(seed))/sqrt_light_count);
			zeroth_btf->LightsParabolic[b*sqrt_light_count + a] = Tempest::CartesianToParabolicCoordinates(dir);
		}

	zeroth_btf->LightCount = light_count;
		
		
	zeroth_btf->UElementStride = zeroth_btf->DataSize,
	zeroth_btf->SxVElementStride = zeroth_btf->DataSize;
	zeroth_btf->Offsets = new uint32_t[2*zeroth_btf->ChannelCount];
	zeroth_btf->ComponentCounts = new uint32_t[zeroth_btf->ChannelCount];

	for(uint32_t chan_idx = 0; chan_idx < zeroth_btf->ChannelCount; ++chan_idx)
	{
		zeroth_btf->Offsets[chan_idx] = 0;
		zeroth_btf->Offsets[zeroth_btf->ChannelCount + chan_idx] = 0;
		zeroth_btf->ComponentCounts[chan_idx] = 1;		
	}
		
	DelaunayTriangulation(zeroth_btf->LightsParabolic, zeroth_btf->LightCount, &zeroth_btf->LightIndices, &zeroth_btf->LightTriangleCount);
	std::unique_ptr<Tempest::LBVH2Node<Tempest::AABB2>[]> light_interm_nodes(Tempest::GenerateTriangleNodes<Tempest::AABB2>(zeroth_btf->LightsParabolic, zeroth_btf->LightCount, zeroth_btf->LightIndices, zeroth_btf->LightTriangleCount));
    zeroth_btf->LightBVH = GenerateSSLBVH(light_interm_nodes.get(), zeroth_btf->LightTriangleCount);

    return zeroth_btf.release();
}

BTF* LoadBTF(const Path& name, BTFExtra* out_extra)
{
	BTFExtra extra;
	BTFPtr btf(new BTF, BTFDeleter());
    std::fstream fs(name.c_str(), std::ios::binary|std::ios::in);

    if(!fs)
    {
        Log(LogLevel::Error, "Failed to load BTF file: ", name);
        return nullptr;
    }

    char c = fs.get();
    if(c != '!')
    {
        Log(LogLevel::Error, "Unsupported BTF format: ", name);
        return nullptr;
    }

    uint32_t signature;
    fs.read(reinterpret_cast<char*>(&signature), sizeof(signature));

    auto fmf1_signature = TEMPEST_MAKE_FOURCC('F', 'M', 'F', '1');

    auto read_common = [&fs, &extra, &btf, &name]()
    {
        bool rotations_included = false;
        auto c = fs.get();
        if(c != 'R')
        {
            fs.unget();
        }
        else
        {
            rotations_included = true;
        }

        auto start_off = fs.tellg();
        fs.read(reinterpret_cast<char*>(&extra.Header), sizeof(extra.Header));

        if(extra.Header.Version > 1)
        {
            fs.read(reinterpret_cast<char*>(&btf->ConsineFlag), sizeof(btf->ConsineFlag));
        }

        if(extra.Header.Version > 2)
        {
            uint32_t len;
            fs.read(reinterpret_cast<char*>(&len), sizeof(len));
            if(len)
            {
                extra.XMLString.resize(len);
                fs.read(&extra.XMLString.front(), len);
            }
        }

        if(extra.Header.Version > 3)
        {
            fs.read(reinterpret_cast<char*>(&btf->ChannelCount), sizeof(btf->ChannelCount));
            extra.Channels.resize(btf->ChannelCount);

            for(uint32_t chan_idx = 0; chan_idx < btf->ChannelCount; ++chan_idx)
            {
                uint32_t chan_size;
                fs.read(reinterpret_cast<char*>(&chan_size), sizeof(chan_size));
                auto& chan = extra.Channels[chan_idx];
                chan.resize(chan_size);
                fs.read(&chan.front(), chan_size);
            }
        }
        else
        {
            extra.Channels = { "R", "G", "B" };
        }

		btf->Offsets = new uint32_t[2*btf->ChannelCount];
		btf->ComponentCounts = new uint32_t[btf->ChannelCount];

        auto header_read = fs.tellg() - start_off;
        if(header_read != extra.Header.Size)
        {
            Log(LogLevel::Error, "Failed to parse BTF header: ", name);
            return false;
        }

        uint32_t view_count;
        fs.read(reinterpret_cast<char*>(&view_count), sizeof(view_count));
        std::unique_ptr<Vector3[]> views(new Vector3[view_count]);
        std::unique_ptr<Vector3[]> lights;

        for(uint32_t view_idx = 0; view_idx < view_count; ++view_idx)
        {
            Tempest::Vector2 angles;
            fs.read(reinterpret_cast<char*>(&angles), sizeof(angles));

            views[view_idx] = SphereToCartesianCoordinates(angles);

            uint32_t num_lights;
            fs.read(reinterpret_cast<char*>(&num_lights), sizeof(num_lights));

            if(btf->LightCount == 0)
            {
                btf->LightCount = num_lights;
                lights = std::unique_ptr<Vector3[]>(new Vector3[btf->LightCount]);
                for(uint32_t light_idx = 0; light_idx < btf->LightCount; ++light_idx)
                {
                    fs.read(reinterpret_cast<char*>(&angles), sizeof(angles));
                    lights[light_idx] = SphereToCartesianCoordinates(angles);
                }
            }
            else if(btf->LightCount == num_lights)
            {
                fs.seekg(sizeof(angles)*btf->LightCount, std::ios::cur);
            }
            else
            {
                Log(LogLevel::Error, "Fixed light hemisphere is only allowed");
                return false;
            }
        }

        fs.read(reinterpret_cast<char*>(&btf->Width), sizeof(btf->Width));
        fs.read(reinterpret_cast<char*>(&btf->Height), sizeof(btf->Height));

        if(rotations_included)
        {
            uint32_t num_rotations;
            fs.read(reinterpret_cast<char*>(&num_rotations), sizeof(num_rotations));

            if(num_rotations)
            {
                extra.Rotations.resize(num_rotations);
                fs.read(reinterpret_cast<char*>(&extra.Rotations.front()), extra.Rotations.size()*sizeof(extra.Rotations.front()));
            }
        }

        btf->LightsParabolic = new Vector2[btf->LightCount];
        for(uint32_t idx = 0; idx < btf->LightCount; ++idx)
        {
            btf->LightsParabolic[idx] = CartesianToParabolicCoordinates(lights[idx]);
        }

        DelaunayTriangulation(btf->LightsParabolic, btf->LightCount, &btf->LightIndices, &btf->LightTriangleCount);
        std::unique_ptr<LBVH2Node<AABB2>[]> light_interm_nodes(GenerateTriangleNodes<AABB2>(btf->LightsParabolic, btf->LightCount, btf->LightIndices, btf->LightTriangleCount));
        btf->LightBVH = GenerateSSLBVH(light_interm_nodes.get(), btf->LightTriangleCount);

        uint32_t edge_count = 3*btf->LightTriangleCount;

        std::unique_ptr<Edge[]> edges(new Edge[3*btf->LightTriangleCount]);
        for(uint32_t tri_idx = 0, tri_end = btf->LightTriangleCount; tri_idx < tri_end; ++tri_idx)
        {
            for(uint32_t edge_idx = 0; edge_idx < 3; ++edge_idx)
            {
                auto& edge = edges[tri_idx*3 + edge_idx];
                edge.Index0 = btf->LightIndices[tri_idx*3 + edge_idx];
                edge.Index1 = btf->LightIndices[tri_idx*3 + (edge_idx + 1) % 3];
                if(edge.Index0 > edge.Index1)
                    std::swap(edge.Index0, edge.Index1);
                edge.Triangle = tri_idx;
            }
        }

        std::sort(edges.get(), edges.get() + edge_count,
                  [](const Edge& lhs, const Edge& rhs)
                  {
                      if(lhs.Index1 != rhs.Index1)
                          return lhs.Index1 < rhs.Index1;
                      if(lhs.Index0 != rhs.Index0)
                          return lhs.Index0 < rhs.Index0;
                      return lhs.Triangle < rhs.Triangle;
                  });

        bool skip_flag = false;
        auto write_iter = edges.get();
        for(auto iter = edges.get(), iter_end = edges.get() + edge_count - 1; iter < iter_end; ++iter)
        {
            auto next_iter = iter + 1;
            if(iter->Index0 == next_iter->Index0 &&
               iter->Index1 == next_iter->Index1)
            {
                skip_flag = true;
                continue;
            }
            else if(skip_flag)
            {
                skip_flag = false;
                continue;
            }
            *write_iter = *iter;
            ++write_iter;
        }

        if(!skip_flag)
        {
            *write_iter = edges[edge_count - 1];
            ++write_iter;
        }
        
        for(auto iter = edges.get(), iter_end = write_iter; iter < iter_end; ++iter)
        {
            auto dir0 = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[iter->Index0]),
                 dir1 = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[iter->Index1]);

            auto angle0 = atan2f(dir0.y, dir0.x),
                 angle1 = atan2f(dir1.y, dir1.x);

            if(angle0 > angle1)
            {
                std::swap(iter->Index0, iter->Index1);
                std::swap(angle0, angle1);
            }

            if(angle1 - angle0 > Tempest::MathPi)
            {
                angle0 += 2.0f*Tempest::MathPi;
                std::swap(iter->Index0, iter->Index1);
                std::swap(angle0, angle1);
            }
        
            iter->Angle0 = angle0;
            iter->Angle1 = angle1;
        }

        std::sort(edges.get(), write_iter,
                  [](const Edge& lhs, const Edge& rhs)
                  {
                      if(lhs.Angle1 != rhs.Angle1)
                          return lhs.Angle1 < rhs.Angle1;
                      if(lhs.Angle0 != rhs.Angle0)
                          return lhs.Angle0 < rhs.Angle0;
                      return lhs.Triangle < rhs.Triangle;
                  });

        edge_count = btf->EdgeCount = write_iter - edges.get();
        btf->Edges = new Edge[edge_count];

        std::copy_n(edges.get(), edge_count, btf->Edges);

    #ifndef NDEBUG
        for(auto iter = edges.get(), iter_end = write_iter - 1; iter < iter_end; ++iter)
        {
            auto next_iter = iter + 1;
            TGE_ASSERT(iter->Angle1 == next_iter->Angle0, "Bad triangulation");
        }
    #endif

        if(!std::equal(views.get(), views.get() + view_count, lights.get()))
        {
            Log(LogLevel::Error, "Non-bidirectional functions are unsupported");
            return false;
        }
        /*
        {
            m_ViewCount = view_count;
            m_ViewsParabolic = std::unique_ptr<Vector2[]>(new Vector2[m_ViewCount]);

            for(uint32_t idx = 0; idx < m_ViewCount; ++idx)
            {
                m_ViewsParabolic[idx] = CartesianToParabolicCoordinates(views[idx]);
            }

            DelaunayTriangulation(m_ViewsParabolic.get(), m_ViewCount, &m_ViewIndices, &m_ViewTriangleCount);
            std::unique_ptr<LBVH2Node[]> view_interm_nodes(GenerateTriangleNodes(m_ViewsParabolic.get(), m_ViewCount, m_ViewIndices, m_ViewTriangleCount));
            m_ViewBVH = std::unique_ptr<SimpleStacklessLBVH2Node[]>(GenerateSSLBVH(view_interm_nodes.get(), m_ViewTriangleCount));
        }
        */

        return true;
    };

    switch(signature)
    {
    case TEMPEST_MAKE_FOURCC('D', 'F', 'M', 'F'):
    {
        bool rotations_included = false, ext_header = false;
        
        const char* expected = "08FC";
        for(uint32_t i = 0; expected[i]; ++i)
            if(expected[i] != (c = fs.get()))
            {
                Log(LogLevel::Error, "Invalid signature");
                return nullptr;
            }

        if(!read_common())
            return nullptr;

        uint32_t num_components;
        fs.read(reinterpret_cast<char*>(&num_components), sizeof(num_components));
        
        fs.read(reinterpret_cast<char*>(&btf->ColorModel), sizeof(btf->ColorModel));
        fs.read(reinterpret_cast<char*>(&btf->ColorMean), sizeof(btf->ColorMean));
        static_assert(sizeof(Tempest::Matrix3) == 3*3*sizeof(float), "Invalid matrix size");
        fs.read(reinterpret_cast<char*>(&btf->ColorTransform), sizeof(btf->ColorTransform));

        btf->DataSize = 0;

        auto light_count = btf->LightCount;
        auto expected_row_count = light_count*light_count;
        auto expected_column_count = btf->Width*btf->Height;

        btf->RowCount = expected_row_count;
        btf->ColumnCount = expected_column_count;

		uint32_t u_plane_offset = 0, sxv_plane_offset = 0;

		struct DeleteSubelements
		{
			uint32_t SubDataCount;
            DeleteSubelements(uint32_t count)
                :   SubDataCount(count) {}
			void operator()(uint8_t** data)
			{
				for(uint32_t i = 0; i < SubDataCount; ++i)
				{
					delete[] data[i];
				}
				delete[] data;
			}
		};

		std::unique_ptr<uint8_t*[], DeleteSubelements> matrix_planes(new uint8_t*[2*btf->ChannelCount], DeleteSubelements(2*btf->ChannelCount));

        for(uint32_t chan_idx = 0, chan_idx_end = btf->ChannelCount; chan_idx < chan_idx_end; ++chan_idx)
        {
            uint8_t scalar_size = fs.get();
            
            if(btf->DataSize == 0)
            {
                btf->DataSize = scalar_size;
            }
            else if(btf->DataSize != scalar_size)
            {
                TGE_ASSERT(false, "Stub");
                Log(LogLevel::Error, "Different sized channels are not supported");
;               return nullptr;
            }

            fs.read(reinterpret_cast<char*>(&num_components), sizeof(num_components));
        
            auto num_component = std::max(1u, std::min(num_components, num_components));

			btf->ComponentCounts[chan_idx] = num_component;

            uint32_t slice_size = num_component*btf->ChannelCount*btf->DataSize;

			auto u_plane = matrix_planes[chan_idx] = new uint8_t[expected_row_count*slice_size];
            auto SxV_plane = matrix_planes[btf->ChannelCount + chan_idx] = new uint8_t[expected_column_count*slice_size];
            
            uint32_t num_row, num_column;
            fs.read(reinterpret_cast<char*>(&num_row), sizeof(num_row));
            fs.read(reinterpret_cast<char*>(&num_column), sizeof(num_column));


            if(num_row != expected_row_count)
            {
                Log(LogLevel::Error, "Invalid BTF row count: ", name);
                return nullptr;
            }

            if(num_column != expected_column_count)
            {
                Log(LogLevel::Error, "Invalid BTF column count: ", name);
                return nullptr;
            }

            size_t scalars_size = num_components*btf->DataSize;
            std::unique_ptr<uint8_t[]> scalars(new uint8_t[scalars_size]);
            fs.read(reinterpret_cast<char*>(scalars.get()), scalars_size);

            if(!fs)
            {
                Log(LogLevel::Error, "Failed to parse BTF file: ", name);
                return nullptr;
            }

			uint32_t left_singular_size = btf->RowCount*num_components*btf->DataSize;
			fs.read(reinterpret_cast<char*>(u_plane), left_singular_size);

			if(!fs)
			{
				Log(LogLevel::Error, "Failed to parse BTF file: ", name);
                return nullptr;
			}

			uint32_t right_singular_size = btf->ColumnCount*num_components*btf->DataSize;
			fs.read(reinterpret_cast<char*>(SxV_plane), right_singular_size);

			if(!fs)
			{
				Log(LogLevel::Error, "Failed to parse BTF file: ", name);
                return nullptr;
			}

			btf->Offsets[chan_idx] = u_plane_offset;
			u_plane_offset += left_singular_size;
			btf->Offsets[btf->ChannelCount + chan_idx] = sxv_plane_offset;
			sxv_plane_offset += right_singular_size;

			/*
            uint32_t piece_size = btf->ComponentCount*btf->DataSize;
            for(uint32_t lv_idx = 0; lv_idx < num_row; ++lv_idx)
            {
                uint8_t* u_slice = btf->LeftSingularU + (lv_idx + chan_idx)*piece_size;
                fs.read(reinterpret_cast<char*>(u_slice), piece_size);
            }

            for(uint32_t xy_idx = 0; xy_idx < num_column; ++xy_idx)
            {
                uint8_t* sxv_slice = btf->RightSingularSxV + (xy_idx + chan_idx)*piece_size;
                fs.read(reinterpret_cast<char*>(sxv_slice), piece_size);
            }
			*/
        }

		btf->LeftSingularUSize = u_plane_offset;
		btf->RightSingularSxVSize = sxv_plane_offset;

		uint32_t plane_idx = 0;
		{
		btf->LeftSingularU = new uint8_t[u_plane_offset];
		uint32_t offset = 0;
		for(uint32_t end_idx = btf->ChannelCount; plane_idx < end_idx; ++plane_idx)
		{
			auto size = btf->RowCount*btf->ComponentCounts[plane_idx]*btf->DataSize;
			memcpy(btf->LeftSingularU + offset, matrix_planes[plane_idx], size);
			offset += size;
		}
		TGE_ASSERT(offset == u_plane_offset, "Invalid data offset");
		}
		
		{
		btf->RightSingularSxV = new uint8_t[sxv_plane_offset];
		uint32_t offset = 0;
		for(uint32_t end_idx = 2*btf->ChannelCount, comp_idx = 0; plane_idx < end_idx; ++plane_idx, ++comp_idx)
		{
			auto size = btf->ColumnCount*btf->ComponentCounts[comp_idx]*btf->DataSize;
			memcpy(btf->RightSingularSxV + offset, matrix_planes[plane_idx], size);
			offset += size;
		}
		TGE_ASSERT(offset == sxv_plane_offset, "Invalid data offset");
		}	

        auto end_pos = fs.tellg();
        fs.seekg(0, std::ios::end);
        auto actual_end_pos = fs.tellg();
        if(end_pos != actual_end_pos)
        {
            Log(LogLevel::Error, "Additional data that cannot be parsed available: ", name);
            return nullptr;
        }

        btf->UElementStride = btf->DataSize;
        btf->SxVElementStride = btf->DataSize;

    } break;
    case TEMPEST_MAKE_FOURCC('F', 'M', 'F', '0'):
    case TEMPEST_MAKE_FOURCC('F', 'M', 'F', '1'):
    {
        bool rotations_included = false, ext_header = false;
        const char* expected;
        if(signature == TEMPEST_MAKE_FOURCC('F', 'M', 'F', '1'))
        {
            ext_header = true;
            expected = "2FCE";
        }
        else
            expected = "6FC";

        for(uint32_t i = 0; expected[i]; ++i)
            if(expected[i] != (c = fs.get()))
            {
                Log(LogLevel::Error, "Invalid signature");
                return nullptr;
            }

        if(!read_common())
            return nullptr;

        if(ext_header)
        {
            char header_version = fs.get();
            if(header_version >= 1)
            {
                fs.read(reinterpret_cast<char*>(&btf->DynamicRangeReduction), sizeof(btf->DynamicRangeReduction));
            }

            if(header_version >= 2)
            {
                fs.read(reinterpret_cast<char*>(&btf->HeightMapSize), sizeof(btf->HeightMapSize));
                if(btf->HeightMapSize.Width && btf->HeightMapSize.Height)
                {
                    uint32_t tex_area = btf->HeightMapSize.Width*btf->HeightMapSize.Height;
                    btf->HeightMap = new uint16_t[tex_area];

                    fs.read(reinterpret_cast<char*>(btf->HeightMap), sizeof(btf->HeightMap[0])*tex_area);
                }
            }
            
            if(header_version >= 3)
            {
                Log(LogLevel::Error, "Unsupported BTF header version: ", header_version, ": ", name);
                return nullptr;
            }
        }

        uint32_t num_components0, num_components1;
        fs.read(reinterpret_cast<char*>(&num_components0), sizeof(num_components0));

        btf->DataSize = fs.get();

        fs.read(reinterpret_cast<char*>(&num_components1), sizeof(num_components1));
        
		auto num_components = std::max(1u, std::min(num_components0, num_components1));
		for(uint32_t i = 0; i < btf->ChannelCount; ++i)
		{
			btf->ComponentCounts[i] = num_components;
		}
        fs.read(reinterpret_cast<char*>(&btf->RowCount), sizeof(btf->RowCount));
        fs.read(reinterpret_cast<char*>(&btf->ColumnCount), sizeof(btf->ColumnCount));

        if(!fs)
        {
            Log(LogLevel::Error, "Failed to parse BTF file: ", name);
            return nullptr;
        }

        size_t scalars_size = num_components*btf->DataSize;
        std::unique_ptr<uint8_t[]> scalars(new uint8_t[scalars_size]);
        fs.read(reinterpret_cast<char*>(scalars.get()), scalars_size);

        if(!fs)
        {
            Log(LogLevel::Error, "Failed to parse BTF file: ", name);
            return nullptr;
        }

        auto light_count = btf->LightCount;
        if(btf->RowCount != light_count*light_count*btf->ChannelCount)
        {
            Log(LogLevel::Error, "Invalid BTF row count: ", name);
            return nullptr;
        }

        if(btf->ColumnCount != btf->Width*btf->Height)
        {
            Log(LogLevel::Error, "Invalid BTF column count: ", name);
            return nullptr;
        }

        size_t left_singular_size = btf->RowCount*num_components*btf->DataSize;
        btf->LeftSingularU = new uint8_t[left_singular_size];
		btf->LeftSingularUSize = left_singular_size;
        fs.read(reinterpret_cast<char*>(btf->LeftSingularU), left_singular_size);

        if(!fs)
        {
            Log(LogLevel::Error, "Failed to parse BTF file: ", name);
            return nullptr;
        }

        size_t right_singular_size = btf->ColumnCount*num_components*btf->DataSize;
		btf->RightSingularSxVSize = right_singular_size;
        btf->RightSingularSxV = new uint8_t[right_singular_size];
        fs.read(reinterpret_cast<char*>(btf->RightSingularSxV), right_singular_size);

        if(!fs)
        {
            Log(LogLevel::Error, "Failed to parse BTF file: ", name);
            return nullptr;
        }

		for(uint32_t idx = 0; idx < btf->ChannelCount; ++idx)
		{
			btf->Offsets[idx] = idx*num_components*btf->DataSize;
		}

		uint32_t offset = btf->ChannelCount;
		for(uint32_t idx = 0; idx < btf->ChannelCount; ++idx)
		{
			btf->Offsets[offset + idx] = 0;
		}

        btf->UElementStride = btf->ChannelCount*btf->DataSize;
        btf->SxVElementStride = btf->DataSize;

        auto end_pos = fs.tellg();
        fs.seekg(0, std::ios::end);
        auto actual_end_pos = fs.tellg();
        if(end_pos != actual_end_pos)
        {
            Log(LogLevel::Error, "Additional data that cannot be parsed available: ", name);
            return nullptr;
        }

        
    } break;
    case TEMPEST_MAKE_FOURCC('P', 'V', 'F', '0'):
    {
        TGE_ASSERT(false, "Stub");
        Log(LogLevel::Error, "Unsupported BTF format: ", name);
        return nullptr;
    } break;
    case TEMPEST_MAKE_FOURCC('B', 'D', 'I', 'F'):
    {
        TGE_ASSERT(false, "Stub");
        Log(LogLevel::Error, "Unsupported BTF format: ", name);
        return nullptr;
    } break;
    }

    // Flip origin
    uint32_t chan_count = btf->ChannelCount;
	uint32_t* sxv_offsets = btf->Offsets + btf->ChannelCount;
    for(uint32_t y = 0, btf_height = btf->Height, y_end = btf_height / 2; y < y_end; ++y)
    {
        for(uint32_t x = 0, btf_width = btf->Width; x < btf_width; ++x)
        {
            uint32_t src_xy_idx = y*btf_width + x;
            uint32_t dst_xy_idx = (btf_height - 1 - y)*btf_width + x;
            for(uint32_t chan_idx = 0; chan_idx < chan_count; ++chan_idx)
            {
		        auto component_count = btf->ComponentCounts[chan_idx];
		        uint32_t src_SxV_elem_offset = src_xy_idx*component_count*btf->SxVElementStride;
                uint32_t dst_SxV_elem_offset = dst_xy_idx*component_count*btf->SxVElementStride;
		        uint32_t SxV_offset = sxv_offsets[chan_idx];
		        auto src_SxVslice = btf->RightSingularSxV + SxV_offset + src_SxV_elem_offset;
                auto dst_SxVslice = btf->RightSingularSxV + SxV_offset + dst_SxV_elem_offset;

                std::swap_ranges(src_SxVslice, src_SxVslice + component_count*btf->DataSize, dst_SxVslice);
            }
        }
    }

	if(out_extra)
	{
		*out_extra = extra;
	}

    return btf.release();
}

void BTFParallelExtractLuminanceSlice(const BTF* btf_cpu, uint32_t id, ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, float** out_luminance_slice)
{
    uint32_t light_count = btf_cpu->LightCount,
             view_count = btf_cpu->LightCount,
             lv_size = light_count*view_count,
             btf_width = btf_cpu->Width;

    auto btf_xy_idx = btf_y*btf_width + btf_x;

    float* lv_lum_slice = *out_luminance_slice;
    auto slice_light = Tempest::CreateParallelForLoop2D(light_count, view_count, 64,
                                                        [btf_cpu, btf_xy_idx, lv_lum_slice](uint32_t worker_id, uint32_t light_idx, uint32_t view_idx) 
        {
            uint32_t lv_idx = view_idx*btf_cpu->LightCount + light_idx;
            auto spec = BTFFetchSpectrum(btf_cpu, lv_idx, btf_xy_idx);
  
            float luminance = Tempest::RGBToLuminance(Tempest::SpectrumToRGB(spec)); // TODO: Luminance out of spectrum
            spec = luminance > 1e-6f ? spec / luminance : Tempest::ToSpectrum(1.0f);

            lv_lum_slice[lv_idx] = luminance;
		});
    pool.enqueueTask(&slice_light);

    pool.waitAndHelp(id, &slice_light);
}

void BTFParallelExtractRGBSlice(const BTF* btf_cpu, uint32_t id, ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, Vector3** out_spectrum_slice)
{
    uint32_t light_count = btf_cpu->LightCount,
             view_count = btf_cpu->LightCount,
             lv_size = light_count*view_count,
             btf_width = btf_cpu->Width;

    auto btf_xy_idx = btf_y*btf_width + btf_x;

    Vector3* lv_spec_slice = *out_spectrum_slice;
    auto slice_light = Tempest::CreateParallelForLoop2D(light_count, view_count, 64,
                                                        [btf_cpu, btf_xy_idx, lv_spec_slice](uint32_t worker_id, uint32_t light_idx, uint32_t view_idx) 
        {
            uint32_t lv_idx = view_idx*btf_cpu->LightCount + light_idx;
            lv_spec_slice[lv_idx] = Tempest::SpectrumToRGB(BTFFetchSpectrum(btf_cpu, lv_idx, btf_xy_idx));
		});
    pool.enqueueTask(&slice_light);

    pool.waitAndHelp(id, &slice_light);
}

void BTFParallelExtractLuminanceSlice(const BTF* btf_cpu, uint32_t id, ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, float** out_luminance_slice, Tempest::Spectrum* out_avg_spec)
{
    uint32_t light_count = btf_cpu->LightCount,
             view_count = btf_cpu->LightCount,
             lv_size = light_count*view_count,
             btf_width = btf_cpu->Width;

    auto thread_count = pool.getThreadCount();
    auto* avg_spec_storage = TGE_TYPED_ALLOCA(Tempest::Spectrum, thread_count);
    memset(avg_spec_storage, 0, thread_count*sizeof(Tempest::Spectrum));

    auto btf_xy_idx = btf_y*btf_width + btf_x;

    float* lv_lum_slice = *out_luminance_slice;
    auto slice_light = Tempest::CreateParallelForLoop2D(light_count, view_count, 64,
                                                        [avg_spec_storage, btf_cpu, btf_xy_idx, lv_lum_slice](uint32_t worker_id, uint32_t light_idx, uint32_t view_idx) 
        {
            uint32_t lv_idx = view_idx*btf_cpu->LightCount + light_idx;
            auto spec = BTFFetchSpectrum(btf_cpu, lv_idx, btf_xy_idx);
  
            float luminance = Tempest::RGBToLuminance(Tempest::SpectrumToRGB(spec)); // TODO: Luminance out of spectrum
            spec = luminance > 1e-6f ? spec / luminance : Tempest::ToSpectrum(1.0f);
                    
            avg_spec_storage[worker_id] += spec;
            lv_lum_slice[lv_idx] = luminance;
		});
    pool.enqueueTask(&slice_light);

    pool.waitAndHelp(id, &slice_light);

    auto avg_spec = avg_spec_storage[0];
    for(uint32_t thread_idx = 1; thread_idx < thread_count; ++thread_idx)
    {
        avg_spec += avg_spec_storage[thread_idx];
    }
    avg_spec /= static_cast<float>(lv_size);

    *out_avg_spec = avg_spec;
}
}
