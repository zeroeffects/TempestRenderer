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

#include "tempest/parser/driver-base.hh"
#include "tempest/utils/logging.hh"

namespace Tempest
{
void DriverBase::warning(const Tempest::Location& loc, const string& str)
{
    ++m_WarningCount;
    Log(LogLevel::Warning, "\n", loc, ": warning: ", str);
}

void DriverBase::warning(const string& str)
{
    ++m_WarningCount;
    Log(LogLevel::Warning, "\n", __FileName, ": warning: ", str);
}

void DriverBase::error(const Tempest::Location& loc, const string& str)
{
    ++m_ErrorCount;
    Log(LogLevel::Error, "\n", loc, ": error: ", str);
}

void DriverBase::error(const string& str)
{
    ++m_ErrorCount;
    Log(LogLevel::Error, "\n", __FileName, ": error: ", str);
}
}