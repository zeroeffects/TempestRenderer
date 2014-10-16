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

#ifndef _TEMPEST_DRIVER_BASE_HH_
#define _TEMPEST_DRIVER_BASE_HH_

#include "tempest/utils/types.hh"

#include <iostream>

namespace Tempest
{
struct Location
{
    std::string* filename;
    size_t       startLine;
    size_t       startColumn;
};    

/*! \brief Convenience macro to point to the default AST::Location used for built-in types and variables
 * 
 *  \remarks It is used internally. Please, don't use this type for anything outside the standard parser,
 *           because it is used for distinguishing built-in types.
 */
#define TGE_DEFAULT_LOCATION Tempest::Location()

inline std::ostream& operator<<(std::ostream& os, const Location& loc)
{
    os << *loc.filename << ":" << loc.startLine << ":" << loc.startColumn;
}

class DriverBase
{
protected:
    size_t                  m_ErrorCount = 0,
                            m_WarningCount = 0;
public:
    DriverBase()=default;
     ~DriverBase()=default;

    string getFileName() const { return __FileName; }

    void warning(const Location& loc, const string& filename);
    void warning(const string& filename);

    void error(const Location& loc, const string& filename);
    void error(const string& filename);
    
    string                  __FileName;
};
}

#endif // _TEMPEST_DRIVER_BASE_HH_