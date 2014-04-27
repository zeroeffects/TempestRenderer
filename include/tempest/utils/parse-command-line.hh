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

#ifndef _PARSE_CMDLINE_HH_
#define _PARSE_CMDLINE_HH_

#include "tempest/utils/assert.hh"

#include <limits>
#include <algorithm>
#include <unordered_map>

namespace Tempest
{
const char FullNameOnly = std::numeric_limits<char>::max();    

class OptionDescription
{
public:
    OptionDescription()
        :   m_ShortName(FullNameOnly),
            m_Arg(false) {}
    
    OptionDescription(char short_name, string description, bool has_arg)
        :   m_ShortName(short_name),
            m_Description(description),
            m_Arg(has_arg) {}
    
    OptionDescription(OptionDescription&& _desc)
        :   m_ShortName(_desc.m_ShortName),
            m_Description(std::move(_desc.m_Description)),
            m_Value(std::move(_desc.m_Value)),
            m_Arg(_desc.m_Arg) {}
    
    OptionDescription& operator=(OptionDescription&& _desc)
    {
        m_ShortName = _desc.m_ShortName;
        m_Description = std::move(_desc.m_Description);
        m_Value = std::move(_desc.m_Value);
        m_Arg = _desc.m_Arg;
        return *this;
    }
    
    OptionDescription(const OptionDescription& _desc)
        :   m_ShortName(_desc.m_ShortName),
            m_Description(_desc.m_Description),
            m_Value(_desc.m_Value),
            m_Arg(_desc.m_Arg) {}
    
    OptionDescription& operator=(const OptionDescription& _desc)
    {
        m_ShortName = _desc.m_ShortName;
        m_Description = _desc.m_Description;
        m_Value = _desc.m_Value;
        m_Arg = _desc.m_Arg;
        return *this;
    }
    
    template<class T>
    T convert()
    {
        TGE_ASSERT(m_Arg, "The requested command line option does not support follow-up values");
        T val;
        std::stringstream ss;
        ss << m_Value;
        ss >> val;
        return val;
    }
    
    void setValue(string value) { m_Value = value; }
    string getValue() const { return m_Value; }
    
    char getShortName() const { return m_ShortName; }
    bool getHasValue() const { return m_Arg; }
    
    string getDescription() const { return m_Description; }
    
private:
    
    
    char   m_ShortName;
    string m_Description;
    string m_Value;
    bool   m_Arg;
};

class CommandLineOptsParser
{
    typedef std::unordered_map<string, OptionDescription> OptionDescriptionMap;
    typedef std::vector<string>                           UntypedValuesList;
    string                     m_Name;
    bool                       m_UnassociatedAllowed;
    OptionDescriptionMap       m_Opts;
    UntypedValuesList          m_Unassociated;
public:
    CommandLineOptsParser(string name, bool unassociated_allowed);
     ~CommandLineOptsParser();
    
    void createOption(char short_name, string full_name, string description, bool has_arg);
    
    template<typename T>
    T extract(const string& full_name)
    {
        auto iter = m_Opts.find(full_name);
        TGE_ASSERT(iter != m_Opts.end(), "Unknown command line option");
        return iter->second.convert<T>();
    }
    
    string getUnassociatedArgument(size_t idx) { return m_Unassociated[idx]; }
    size_t getUnassociatedCount() const { return m_Unassociated.size(); }
    
    bool parse(int argc, char* argv[]);
    
    void printHelp(std::ostream& os);
};
}

#endif // _PARSE_CMDLINE_HH_