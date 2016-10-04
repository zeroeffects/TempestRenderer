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

#include "tempest/utils/patterns.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/assert.hh"

#include <cstdlib>

void _TestCaseFunction();

namespace Tempest
{
int TestingEnvironment(const char* desc)
{
    try
    {
        Tempest::Log(Tempest::LogLevel::Info, "TEST: ", desc);
        _TestCaseFunction();
    }
    catch(const std::exception& e)
    {
        Tempest::Log(Tempest::LogLevel::Fatal, "Standard exception captured:\n\t", e.what());
        return EXIT_FAILURE;
    } 
    catch(...)
    {
        Tempest::Log(Tempest::LogLevel::Fatal, "Unknown exception captured");
        return EXIT_FAILURE;
    }
    Tempest::Log(Tempest::LogLevel::Info, "SUCCESSFULLY COMPLETED!");\
    return EXIT_SUCCESS;
}
}

#define TGE_TEST(desc) \
    int TempestMain(int argc, char** argv) {\
        return Tempest::TestingEnvironment(desc); \
    } \
    void _TestCaseFunction()

//! \brief This macro does the same as TGE_ASSERT; however, it does not get optimized out in release builds
#define TGE_CHECK(statement, doc_msg) _TGE_ASSERT(statement, doc_msg, CONCAT_MACRO(__ignoreAssert, __COUNTER__))
