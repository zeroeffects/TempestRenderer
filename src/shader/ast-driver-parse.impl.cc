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

bool Driver::parseFile(const string& filename)
{
    // Ye olde version of FLEX on Windows does not support this function.
    #ifndef _WIN32
    //auto del_scanner = CreateAtScopeExit([]() { yylex_destroy(); });
    #endif

    Parser parser(*this);
    __FileName = filename;
    /*
    #ifndef NDEBUG
    yy_flex_debug = true;
    #endif
    parser.set_debug_level(yy_flex_debug);
    */
    yyin = fopen(filename.c_str(), "rt");
    auto fd = CreateAtScopeExit([]() {
                                    if(yyin)
                                        fclose(yyin);
                                });

    if(!yyin)
    {
        Tempest::Log(LogLevel::Error, "Error has occurred while trying to open the following file: ", filename);
        return false;
    }

    int res = parser.parse();

    return res == 0 && !m_ErrorCount;
}

bool Driver::parseString(const char* str, size_t size, const string& filename)
{
    // Ye olde version of FLEX on Windows does not support this function.
    #ifndef _WIN32
    //auto del_scanner = CreateAtScopeExit([]() { yylex_destroy(); });
    #endif

    Parser parser(*this);
    __FileName = filename;
    /*
    #ifndef NDEBUG
    yy_flex_debug = true;
    #endif
    parser.set_debug_level(yy_flex_debug);
    */
    auto bs = shader__scan_bytes(str, size);
    auto fd = CreateAtScopeExit([bs]() { shader__delete_buffer(bs); });

    if(!bs)
    {
        Tempest::Log(LogLevel::Error, "Cannot create buffer object for file: ", filename);
        return false;
    }

    shader__switch_to_buffer(bs);
    
    int res = parser.parse();

    return res == 0 && !m_ErrorCount;
}