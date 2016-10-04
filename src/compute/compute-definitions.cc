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

#include "tempest/compute/compute-definitions.hh"

#include "channel_descriptor.h"

namespace Tempest
{
cudaChannelFormatDesc CUDAChannelFormats[(size_t)DataFormat::Count]
{
    cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone),         // Unknown,
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat),       // R32F,
    cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat),      // RG32F,
    cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat),     // RGB32F,
    cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat),    // RGBA32F,
    cudaCreateChannelDescHalf(),                                          // R16F,
    cudaCreateChannelDescHalf2(),                                         // RG16F,
//  RGB16F,
    cudaCreateChannelDescHalf4(),                                         // RGBA16F,
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned),      // R32,
    cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned),     // RG32,
    cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindSigned),    // RGB32,
    cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned),   // RGBA32,
    cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned),      // R16,
    cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSigned),     // RG16,
//  RGB16,
    cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSigned),   // RGBA16,
    cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSigned),       // R8,
    cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSigned),       // RG8,
//  RGB8,
    cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSigned),       // RGBA8,
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned),    // uR32,
    cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned),   // uRG32,
    cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindUnsigned),  // uRGB32,
    cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned), // uRGBA32,
    cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned),    // uR16,
    cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned),   // uRG16,
//  uRGB16,
    cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned), // uRGBA16,
    cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned),     // uR8,
    cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned),     // uRG8,
//  uRGB8,
    cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned),     // uRGBA8,
    
    cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned),      // R16SNorm,
    cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSigned),     // RG16SNorm,
//  RGB16SNorm,
    cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSigned),   // RGBA16SNorm,
    cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSigned),       // R8SNorm,
    cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSigned),       // RG8SNorm,
//  RGB8SNorm,
    cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSigned),       // RGBA8SNorm,
    cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned),    // R16UNorm,
    cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned),   // RG16UNorm,
//  RGB16UNorm,
    cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned), // RGBA16UNorm,
    cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned),     // R8UNorm,
    cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned),     // RG8UNorm,
//  RGB8UNorm,
    cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned),    // RGBA8UNorm,

    cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned),   // D16,
    cudaCreateChannelDesc(24, 8, 0, 0, cudaChannelFormatKindUnsigned),   // D24S8, - well, probably no
    cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned),   // D32,

    cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindSigned),   // R10G10B10A2,
    cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsigned), // uR10G10B10A2,
};
}