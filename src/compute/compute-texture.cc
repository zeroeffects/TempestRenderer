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

#include "tempest/compute/compute-texture.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/graphics/cube-map.hh"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace Tempest
{
cudaTextureObject_t CreateCudaTexture(const Texture* tex, uint32_t flags)
{
    cudaError err;
    void* cuda_tex_mem;
    size_t pitch;
    auto& hdr = tex->getHeader();
    uint32_t width = hdr.Width,
             height = hdr.Height,
             el_size = DataFormatElementSize(hdr.Format),
             spitch = el_size*width;

    auto chan_desc = DataFormatToCuda(hdr.Format);

    cudaTextureObject_t tex_obj;
    cudaResourceDesc res_desc = {};
    cudaTextureDesc tex_desc = {};
    if(flags & TEMPEST_CUDA_TEXTURE_GATHER)
    {
        err = cudaMallocPitch(&cuda_tex_mem, &pitch, spitch, height);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "Failed to allocate cuda memory for texture");
            return 0;
        }

        err = cudaMemcpy2D(cuda_tex_mem, pitch, tex->getData(), spitch, spitch, height, cudaMemcpyHostToDevice);
        if(err != cudaSuccess)
        {
            cudaFree(cuda_tex_mem);
            Log(LogLevel::Error, "Failed to copy texture to GPU memory");
            return 0;
        }

        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = cuda_tex_mem;
        res_desc.res.pitch2D.pitchInBytes = pitch;
        res_desc.res.pitch2D.width = width;
        res_desc.res.pitch2D.height = height;
        res_desc.res.pitch2D.desc = chan_desc;
    }
    else
    {
        cudaArray_t array;

        auto err = cudaMallocArray(&array, &chan_desc, hdr.Width, hdr.Height, cudaArrayTextureGather);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "Failed to create surface: failed to allocate memory: ", cudaGetErrorString(err));
		    return 0;
        }

        size_t pitch = hdr.Width*DataFormatElementSize(hdr.Format);
        err = cudaMemcpy2DToArray(array, 0, 0, tex->getData(), pitch, pitch, hdr.Height, cudaMemcpyHostToDevice);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "Failed to copy surface data: ", cudaGetErrorString(err));
		    cudaFreeArray(array);
            return 0;
        }

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = array;
    }

    tex_desc.addressMode[0] = tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = TranslateFilterMode(hdr.Sampling);
    tex_desc.maxAnisotropy = 1;
    tex_desc.normalizedCoords = true;
    tex_desc.sRGB = (flags & TEMPEST_CUDA_TEXTURE_SRGB) != 0;
    tex_desc.readMode = TranslateReadMode(hdr.Format);

    err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
    if(err != cudaSuccess)
    {
        if(res_desc.resType == cudaResourceTypePitch2D)
        {
            cudaFree(res_desc.res.pitch2D.devPtr);
        }
        else if(res_desc.resType == cudaResourceTypeArray)
        {
            cudaFree(res_desc.res.array.array);
        }
        Log(LogLevel::Error, "Failed to bind texture to cuda ray tracer pipeline");
        return 0;
    }

    return tex_obj;
}

cudaTextureObject_t CreateCudaTexture(const CubeMap* tex, uint32_t flags)
{
    cudaError err;
    auto& hdr = tex->getHeader();
    uint32_t width = hdr.Width,
             height = hdr.Height,
             depth = hdr.Depth*6,
             el_size = DataFormatElementSize(hdr.Format),
             spitch = el_size*width;

    auto chan_desc = DataFormatToCuda(hdr.Format);

    cudaTextureObject_t tex_obj;
    cudaResourceDesc res_desc = {};
    cudaResourceViewDesc res_view = {};
    cudaTextureDesc tex_desc = {};

    auto src_pitch = hdr.Width*el_size;
    cudaExtent extent{ hdr.Width, hdr.Height, depth };

    if(1)
    {
        cudaArray_t cuda_array;
        err = cudaMalloc3DArray(&cuda_array, &chan_desc, extent, cudaArrayCubemap);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "Failed to allocate cuda memory for texture");
            return 0;
        }

        cudaMemcpy3DParms memcpy_params;
        memcpy_params.srcArray = 0;
        memcpy_params.srcPos = {};
        memcpy_params.srcPtr = cudaPitchedPtr{ const_cast<void*>(tex->getData()), src_pitch, src_pitch, hdr.Height };
  
        memcpy_params.dstArray = cuda_array;
        memcpy_params.dstPos = {};
        memcpy_params.dstPtr = {};
  
        memcpy_params.extent = extent;
        memcpy_params.kind = cudaMemcpyHostToDevice;
        err = cudaMemcpy3D(&memcpy_params);
        if(err != cudaSuccess)
        {
            cudaFreeArray(cuda_array);
            Log(LogLevel::Error, "Failed to copy texture to GPU memory");
            return 0;
        }

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;
    }
    else
    {
        cudaPitchedPtr pitched_ptr;
        err = cudaMalloc3D(&pitched_ptr, extent);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "Failed to allocate cuda memory for texture");
            return 0;
        }

        cudaMemcpy3DParms memcpy_params;
        memcpy_params.srcArray = 0;
        memcpy_params.srcPos = {};
        memcpy_params.srcPtr = cudaPitchedPtr{ const_cast<void*>(tex->getData()), src_pitch, src_pitch, hdr.Height };
  
        memcpy_params.dstArray = 0;
        memcpy_params.dstPos = {};
        memcpy_params.dstPtr = pitched_ptr;
  
        memcpy_params.extent = extent;
        memcpy_params.kind = cudaMemcpyHostToDevice;
        err = cudaMemcpy3D(&memcpy_params);
        if(err != cudaSuccess)
        {
            cudaFree(pitched_ptr.ptr);
            Log(LogLevel::Error, "Failed to copy texture to GPU memory");
            return 0;
        }

        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = pitched_ptr.ptr;
        res_desc.res.pitch2D.pitchInBytes = pitched_ptr.pitch;
        res_desc.res.pitch2D.width = width;
        res_desc.res.pitch2D.height = height;
        res_desc.res.pitch2D.desc = chan_desc;
    }

    
    tex_desc.addressMode[0] = tex_desc.addressMode[1] = tex_desc.addressMode[2] = cudaAddressModeWrap;
    tex_desc.filterMode = TranslateFilterMode(hdr.Sampling);
    tex_desc.maxAnisotropy = 1;
    tex_desc.normalizedCoords = true;
    tex_desc.sRGB = (flags & TEMPEST_CUDA_TEXTURE_SRGB) != 0;
    tex_desc.readMode = TranslateReadMode(hdr.Format);

    err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
    if(err != cudaSuccess)
    {
        if(res_desc.resType == cudaResourceTypePitch2D)
        {
            cudaFree(res_desc.res.pitch2D.devPtr);
        }
        else if(res_desc.resType == cudaResourceTypeArray)
        {
            cudaFree(res_desc.res.array.array);
        }
        Log(LogLevel::Error, "Failed to bind texture to cuda ray tracer pipeline");
        return 0;
    }

    return tex_obj;
}

void CudaTextureDeleter(cudaTextureObject_t cuda_tex)
{
    cudaResourceDesc res_desc;
    cudaGetTextureObjectResourceDesc(&res_desc, cuda_tex);
    cudaDestroyTextureObject(cuda_tex);
    if(res_desc.resType == cudaResourceTypePitch2D)
    {
        cudaFree(res_desc.res.pitch2D.devPtr);
    }
    else if(res_desc.resType == cudaResourceTypeArray)
    {
        cudaFree(res_desc.res.array.array);
    }
    else
    {
        TGE_ASSERT(false, "Unsupported type of resource");
    }
}
}