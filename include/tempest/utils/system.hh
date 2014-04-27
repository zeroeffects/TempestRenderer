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

#ifndef _TEMPEST_SYSTEM_HH_
#define _TEMPEST_SYSTEM_HH_

#include "tempest/utils/types.hh"

namespace Tempest
{
namespace System
{
    /*! \brief Gets the value of the specified Environment Variable.
     *  \param name     the name of the Environment Variable.
     *  \return on success it returns true.
     */
    bool GetEnvironmentVariable(const string& name, string& res);

    /*! \brief Sets the value of the specified Environment Variable.
     *  \param name     the name of the Environment Variable.
     *  \param val      the value to which it must be set.
     *  \return on success it returns true.
     */
    bool SetEnvironmentVariable(const string& name, const string& val);

    /*! \brief Gets the full path to the current executable.
     *  \remarks the returned string is in native file format.
     *  \return on failure returns an empty string.
     */
    string GetExecutablePath();

    //! Returns whether the specified file exists.
    bool Exists(const string& name);
}
}

#endif /* _TEMPEST_SYSTEM_HH_ */