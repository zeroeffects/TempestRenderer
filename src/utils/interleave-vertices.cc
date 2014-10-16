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

#include "tempest/utils/interleave-vertices.hh"
#include "tempest/utils/patterns.hh"
#include "xxhash/xxhash.h"

#include <unordered_map>
#include <algorithm>
#include <numeric> 
#include <memory>

namespace Tempest
{
class IndicesHasher
{
    size_t m_ElementCount;
public:
    IndicesHasher(size_t el_count)
        :   m_ElementCount(el_count) {}

    size_t operator()(const int32* key) const
    {
        return XXH32(key, m_ElementCount*sizeof(int32), 0xEF1C1337);
    }
};

class IndicesEquality
{
    size_t m_ElementCount;
public:
    IndicesEquality(size_t el_count)
        :   m_ElementCount(el_count) {}
    
    bool operator()(const int32* key1, const int32* key2) const
    {
        return std::equal(key1, key1 + m_ElementCount, key2);
    }
};

#define ESTIMATED_COMPRESSION_RATIO 2

void InterleaveVertices(const char** vert_arrays,
                        const int32* strides,
                        int32 subarrays,
                        const int32** inds,
                        size_t ind_count,
                        std::vector<int32>* out_inds,
                        std::vector<char>* out_data)
{
    out_data->reserve(std::accumulate(strides, strides + subarrays, 0)*ind_count/ESTIMATED_COMPRESSION_RATIO); // pessimistic, but better than reallocating
    std::unordered_map<int32*, int32, IndicesHasher, IndicesEquality> inds_remap(ind_count, IndicesHasher(subarrays), IndicesEquality(subarrays));
    auto scope_exit = CreateAtScopeExit([&inds_remap]()
                                        {
                                            for(auto& p : inds_remap)
                                            {
                                                delete p.first;
                                            }
                                        });
    std::unique_ptr<int32[]> current_indices(new int32[subarrays]);
    
    int32 index = 0;
    
    for(size_t i = 0; i < ind_count; ++i)
    {
        for(size_t j = 0; j < subarrays; ++j)
        {
            current_indices[j] = inds[j][i];
        }
        auto iter = inds_remap.find(current_indices.get());
        if(iter != inds_remap.end())
        {
            out_inds->push_back(iter->second);
        }
        else
        {
            for(size_t j = 0; j < subarrays; ++j)
            {
                size_t stride = strides[j];
                auto* start_ptr = vert_arrays[j] + stride*current_indices[j];
                auto* end_ptr = start_ptr + stride;
                out_data->insert(out_data->end(), start_ptr, end_ptr);
            }
            out_inds->push_back(index);
            inds_remap[current_indices.release()] = index;
            current_indices = std::unique_ptr<int32[]>(new int32[subarrays]);
            ++index;
        }
    }
        
}
}