/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2010-2014 Zdravko Velinov
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

#include "tempest/utils/assert.hh"

#include <algorithm>

namespace Tempest
{
namespace Graphics
{
struct StateObject;
struct CommandBuffer;

struct PipelineStage
{
    StateObject*   states;
    CommandBuffer* batches;
};

class Pipeline
{
    std::vector<PipelineStage> m_Stages;
public:
    Pipeline() {}
     ~Pipeline() {}
    
    template<class TFunc>
    void applyOnStages(TFunc func) { std::for_each(std::begin(m_Stages), std::end(m_Stages), func); }
    
    template<class TFunc>
    void applyOnStage(size_t idx, TFunc func) { TGE_ASSERT(idx < m_Stages.size(), "Unknown stage"); func(m_Stages[idx]); }
};
}
}